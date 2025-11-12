# input_file是一个data.jsonl文件，数据每一行是这样的
# {"mi_idx": 5, "state": {"receiving_rate": 420295.5665024631, "num_received_packets": 13, "received_bytes": 10665, "queuing_delay": 41.0, "delay_minus_base": 1758606527988.0, "min_seen_delay": 1758606528147, "delay_ratio": 1.0000000000011373, "delay_avg_min_diff": 2.0, "mean_interarrival": 16.916666666666668, "packet_jitter": 13.493825748590847, "packet_loss_ratio": 0.9980983031012288, "avg_lost_pkts": 3411.5, "video_prob": 1.0, "audio_prob": 0.0, "probe_prob": 0.0, "received_video_bytes": 10333, "received_audio_bytes": 0, "payload_type": [125, 125, 125, 125, 125, 125, 125, 125, 122, 125, 125, 125, 125], "send_time": [46968, 46999, 47035, 47035, 47040, 47066, 47102, 47107, 47112, 47133, 47133, 47164, 47164], "receive_time": [1758606575154, 1758606575186, 1758606575222, 1758606575222, 1758606575227, 1758606575253, 1758606575289, 1758606575294, 1758606575299, 1758606575320, 1758606575328, 1758606575351, 1758606575357]}, "action": {"bandwidth_estimation": 578163.8461538461}}

# 现在我要统计指标，按照带宽预测策略和TC的组合，每一组分开统计
# 算receiving_rate和tc_rates_truncated的MSE（rate的单位取Mbps）
# 平均packet_loss_ratio
# 平均delay_avg_min_diff
# 平均tc_losses_truncated
# 平均tc_delays_truncated

# 对齐data.jsonl里指标和tc_profile里指标的时间时
# 两边得共享data.jsonl的时间轴（可以参考plot_trace_tc.py里对齐时间轴的做法）
# 注意data.jsonl里
# send_time是send端发包的时间
# receive_time是receive端收到包的时间
# send端的时间和receive端的时间不是一个时钟
# 但是send_time和receive_time是一一对应的

# tc_profiles文件夹（/home/min414/data2/BoB_MIN/tc_profiles）下有很多文件，文件里是tc配置信息
# 两行为一组，然后文件尾部有空行，例如
# rate 1600kbit
# delay 20ms
# loss 0%
# wait 10s
# rate 400kbit
# delay 250ms
# loss 0%
# wait 250ms
# 比如上面这样，就是rate 1600kbit的真实带宽，持续20ms + 10s
# 直到下一次改rate， rate 400kbit，持续250ms + 250ms
# 如果时间还没到，那么就回到文件最上面的配置，rate 1600kbit，持续20ms + 10s
# 以此类推，循环往复，tc加的丢包率和延迟也是类似的
# 具体解析tc_profile文件，参考tc_policy_min.sh脚本

# input_path下有好多文件夹，表示一条条trace
# 每个文件夹下都有一个data.jsonl
# input_file就是这个data.jsonl
# plot_file就放在data.jsonl同一个路径下
# input_path同路径下有个名字类似webrtc_receive_bob1_bowing_cif_test_30_bad4G.log的文件
# 提取log文件名中的tc_profile_name，比如上面这个例子，tc_profile_name就是bad4G
# 去查找 tc_profiles文件夹下的bad4G文件作为真实带宽变化的依据
# 提取log文件名中的bandwidth_estimation策略，比如上面这个例子，bandwidth_estimation就是bob1

# 解析input_path，可以参考plot_trace_tc.py脚本

import os
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

TC_PROFILE_DIR = '/home/min414/data2/BoB_MIN/tc_profiles'

# ... (parse_unit, parse_tc_profile, get_tc_params_over_time, get_tc_profile_and_strategy 函数保持不变) ...
def parse_unit(value_str):
    value_str = str(value_str).lower().strip()
    if 'kbit' in value_str:
        return float(value_str.replace('kbit', '')) * 1000
    if 'mbit' in value_str:
        return float(value_str.replace('mbit', '')) * 1000000
    if 'ms' in value_str:
        return float(value_str.replace('ms', '')) / 1000
    if 's' in value_str:
        return float(value_str.replace('s', ''))
    if '%' in value_str:
        return float(value_str.replace('%', '')) / 100.0
    try:
        return float(value_str)
    except (ValueError, TypeError):
        return 0.0

def parse_tc_profile(profile_path):
    if not os.path.exists(profile_path):
        return []
    with open(profile_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    if not lines: return []
    rate_groups = []
    current_group = []
    for line in lines:
        if line.startswith('rate'):
            if current_group: rate_groups.append(current_group)
            current_group = [line]
        else:
            current_group.append(line)
    if current_group: rate_groups.append(current_group)
    commands = []
    for group in rate_groups:
        params = {'rate': 0, 'loss': 0, 'delay': 0, 'duration': 0}
        for cmd in group:
            parts = cmd.split()
            if len(parts) < 2: continue
            cmd_type = parts[0]
            cmd_value = parts[1]
            if cmd_type == 'rate':
                params['rate'] = parse_unit(cmd_value)
            elif cmd_type == 'loss':
                params['loss'] = parse_unit(cmd_value)
            elif cmd_type == 'delay':
                params['delay'] = parse_unit(cmd_value)
            elif cmd_type == 'wait':
                params['duration'] += parse_unit(cmd_value)
        commands.append(params)
    return commands

def get_tc_params_over_time(commands, times_sec):
    if not commands or not times_sec:
        n = len(times_sec)
        return [0]*n, [0]*n, [0]*n
    total_cycle = sum(cmd['duration'] for cmd in commands)
    tc_rates, tc_losses, tc_delays = [], [], []
    for t in times_sec:
        t_mod = t % total_cycle if total_cycle > 0 else 0
        acc = 0
        for cmd in commands:
            acc += cmd['duration']
            if t_mod < acc:
                tc_rates.append(cmd['rate'])
                tc_losses.append(cmd['loss'])
                tc_delays.append(cmd['delay'])
                break
    return tc_rates, tc_losses, tc_delays

def get_tc_profile_and_strategy(folder_path):
    for f in os.listdir(folder_path):
        if f.endswith('.log'):
            base_name = os.path.splitext(f)[0]
            parts = base_name.split('_')
            if len(parts) > 2:
                return parts[-1], parts[2]
    return None, None

def process_one_trace(folder_path, use_tqdm=False):
    """处理单个trace文件夹并返回指标字典。"""
    data_file = os.path.join(folder_path, 'data.jsonl')
    if not os.path.isfile(data_file):
        return None
    tc_profile_name, strategy = get_tc_profile_and_strategy(folder_path)
    if not tc_profile_name or not strategy:
        return None
    profile_path = os.path.join(TC_PROFILE_DIR, tc_profile_name)
    commands = parse_tc_profile(profile_path)
    
    times, receiving_rates, packet_loss_ratios, delay_avg_min_diffs = [], [], [], []
    try:
        with open(data_file, 'r') as f:
            lines = f.readlines()
        
        line_iterator = tqdm(lines, desc=f"Parsing {os.path.basename(folder_path)}", leave=False) if use_tqdm else lines
        for line in line_iterator:
            try:
                d = json.loads(line)
                state = d.get('state', {})
                receive_times = state.get('receive_time')
                if not receive_times or not isinstance(receive_times, list): continue
                times.append(receive_times[-1])
                receiving_rates.append(float(state.get('receiving_rate', 0)))
                packet_loss_ratios.append(float(state.get('packet_loss_ratio', 0)))
                delay_avg_min_diffs.append(float(state.get('delay_avg_min_diff', 0)))
            except Exception:
                continue
    except Exception:
        return None

    if not times:
        return None
        
    base_time = times[0]
    times_sec = [(t - base_time) / 1000.0 for t in times]
    tc_rates, tc_losses, tc_delays = get_tc_params_over_time(commands, times_sec)
    
    receiving_rates_mbps = np.array(receiving_rates) / 1e6
    tc_rates_mbps = np.array(tc_rates) / 1e6
    
    mse = np.mean((receiving_rates_mbps - tc_rates_mbps) ** 2)
    avg_loss = np.mean(packet_loss_ratios)
    avg_delay = np.mean(delay_avg_min_diffs)
    avg_tc_loss = np.mean(tc_losses)
    avg_tc_delay = np.mean(tc_delays)
    avg_receiving_rate = np.mean(receiving_rates_mbps)
    
    return {
        'strategy': strategy,
        'tc_profile': tc_profile_name,
        'mse': mse,
        'avg_loss': avg_loss,
        'avg_delay': avg_delay,
        'avg_tc_loss': avg_tc_loss,
        'avg_tc_delay': avg_tc_delay,
        'avg_receiving_rate': avg_receiving_rate,
    }

def calculate_metrics_for_collection(collection_path):
    """
    遍历一个集合文件夹中的所有trace，按 (strategy, tc_profile) 分组计算并返回平均性能指标。
    :param collection_path: 包含多个trace子文件夹的路径。
    :return: 一个字典，键为 'strategy_tc_profile'，值为包含平均指标的字典。
    """
    if not os.path.isdir(collection_path):
        print(f"Warning: Collection path not found: {collection_path}")
        return None

    all_trace_folders = [f for f in os.listdir(collection_path) if os.path.isdir(os.path.join(collection_path, f))]
    
    if not all_trace_folders:
        print(f"Warning: No trace folders found in {collection_path}")
        return None

    # 使用 defaultdict 来按 (strategy, tc_profile) 分组
    grouped_results = defaultdict(list)
    for folder in all_trace_folders:
        folder_path = os.path.join(collection_path, folder)
        res = process_one_trace(folder_path, use_tqdm=False)
        if res:
            # 创建分组的键
            key = (res['strategy'], res['tc_profile'])
            grouped_results[key].append(res)

    if not grouped_results:
        print(f"Warning: No valid traces could be processed in {collection_path}")
        return None

    # 计算每个组的平均指标
    final_metrics = {}
    for (strategy, tc_profile), results_list in grouped_results.items():
        df = pd.DataFrame(results_list)
        mean_values = df.select_dtypes(include=np.number).mean().to_dict()
        
        # 使用 "strategy_tc_profile" 作为最终的键
        final_key = f"{strategy}_{tc_profile}"
        final_metrics[final_key] = mean_values
    
    return final_metrics

def main(input_path, report_file):
    # ... (main 函数保持不变，它用于独立的报告生成) ...
    results = defaultdict(list)
    folders = [f for f in sorted(os.listdir(input_path)) if os.path.isdir(os.path.join(input_path, f))]
    with tqdm(total=len(folders), desc=f"Traces in {input_path}") as pbar:
        for folder in folders:
            folder_path = os.path.join(input_path, folder)
            # 在这里调用时，可以保留tqdm，因为它是在脚本独立运行时的主循环
            res = process_one_trace(folder_path, use_tqdm=True)
            if res:
                key = (res['strategy'], res['tc_profile'])
                results[key].append(res)
            pbar.update(1)
            
    with open(report_file, 'a') as fout:
        fout.write(f"\n==== Results for {input_path} ====\n")
        fout.write("strategy,tc_profile,mse,avg_loss,avg_delay,avg_tc_loss,avg_tc_delay,avg_receiving_rate\n")
        for key, group in results.items():
            strategy, tc_profile = key
            df = pd.DataFrame(group)
            mean_values = df.select_dtypes(include=np.number).mean()
            fout.write(f"{strategy},{tc_profile}," + ",".join([f"{v:.6f}" for v in mean_values.values]) + "\n")

if __name__ == '__main__':
    # ... (主程序入口保持不变) ...
    basedir = '/home/min414/data2/extra_storage'
    ids = ['0','1','2','3','4','5',]
    report_file = 'report.txt'
    with open(report_file, 'w') as fout:
        fout.write('')
    for test_id in ids:
        input_path = os.path.join(basedir, test_id)
        main(input_path, report_file)
