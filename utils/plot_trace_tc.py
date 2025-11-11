# input_file是一个data.jsonl文件，数据每一行是这样的
# {"mi_idx": 5, "state": {"receiving_rate": 420295.5665024631, "num_received_packets": 13, "received_bytes": 10665, "queuing_delay": 41.0, "delay_minus_base": 1758606527988.0, "min_seen_delay": 1758606528147, "delay_ratio": 1.0000000000011373, "delay_avg_min_diff": 2.0, "mean_interarrival": 16.916666666666668, "packet_jitter": 13.493825748590847, "packet_loss_ratio": 0.9980983031012288, "avg_lost_pkts": 3411.5, "video_prob": 1.0, "audio_prob": 0.0, "probe_prob": 0.0, "received_video_bytes": 10333, "received_audio_bytes": 0, "payload_type": [125, 125, 125, 125, 125, 125, 125, 125, 122, 125, 125, 125, 125], "send_time": [46968, 46999, 47035, 47035, 47040, 47066, 47102, 47107, 47112, 47133, 47133, 47164, 47164], "receive_time": [1758606575154, 1758606575186, 1758606575222, 1758606575222, 1758606575227, 1758606575253, 1758606575289, 1758606575294, 1758606575299, 1758606575320, 1758606575328, 1758606575351, 1758606575357]}, "action": {"bandwidth_estimation": 578163.8461538461}}

# 现在我要画图
# 画receiving_rate、bandwidth_estimation、packet_loss_ratio、delay_avg_min_diff随时间变化
# x轴是时间，y轴是值
# 注意send_time是send端发包的时间
# receive_time是receive端收到包的时间
# send端的时间和receive端的时间不是一个时钟
# 但是send_time和receive_time是一一对应的

# 如果TC为True的话，还要画出真实带宽随时间的变化、tc加的丢包率、tc加的延迟随时间的变化（tc加的丢包率和延迟初始值为0）
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

# 然后receiving_rate、bandwidth_estimation、真实带宽放一张子图
# packet_loss_ratio放另一张子图
# delay_avg_min_diff放另一张子图
# tc加的丢包率放另一张子图
# tc加的延迟放另一张子图
# 几张图，共享data.jsonl的时间轴，然后上下排列

# input_path下有好多文件夹，表示一条条trace
# 每个文件夹下都有一个data.jsonl
# input_file就是这个data.jsonl
# plot_file就放在data.jsonl同一个路径下
# input_path同路径下有个名字类似webrtc_receive_bob1_bowing_cif_test_30_bad4G.log的文件
# 提取log文件名中的tc_profile_name，比如上面这个例子，tc_profile_name就是bad4G
# 去查找 tc_profiles文件夹下的bad4G文件作为真实带宽变化的依据

import os
import json
import numpy as np
import sys
import re

# 导入绘图库
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 全局配置 ---
TC = True
TC_PROFILE_DIR = '/home/min414/data2/BoB_MIN/tc_profiles'

def parse_unit(value_str):
    """解析带有单位的字符串，如 '1600kbit', '10s', '250ms'"""
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
    """解析tc profile文件，提取rate, loss, delay, 和 duration"""
    if not os.path.exists(profile_path):
        print(f"  [Warning] TC profile not found: {profile_path}")
        return None
    with open(profile_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    if not lines: return []
    
    commands = []
    rate_groups = []
    current_group = []
    for line in lines:
        if line.startswith('rate'):
            if current_group: rate_groups.append(current_group)
            current_group = [line]
        else:
            current_group.append(line)
    if current_group: rate_groups.append(current_group)

    for group in rate_groups:
        # 默认值
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

def get_tc_params_over_time(commands, total_duration):
    """根据解析的指令和总时长，生成循环的TC参数（rate, loss, delay）序列"""
    if not commands: return [], [], [], []
    
    times, rates, losses, delays = [0], [commands[0]['rate']], [commands[0]['loss']], [commands[0]['delay']]
    current_time = 0
    cmd_idx = 0
    total_cycle_duration = sum(cmd['duration'] for cmd in commands)
    if total_cycle_duration <= 0: 
        return [0, total_duration], [rates[0], rates[0]], [losses[0], losses[0]], [delays[0], delays[0]]

    while current_time < total_duration:
        command = commands[cmd_idx % len(commands)]
        
        if current_time > 0:
            times.append(current_time)
            rates.append(rates[-1])
            losses.append(losses[-1])
            delays.append(delays[-1])
            
        times.append(current_time)
        rates.append(command['rate'])
        losses.append(command['loss'])
        delays.append(command['delay'])
        
        current_time += command['duration']
        
        times.append(current_time)
        rates.append(command['rate'])
        losses.append(command['loss'])
        delays.append(command['delay'])
        
        cmd_idx += 1
        
    if times[-1] < total_duration:
        times.append(total_duration)
        rates.append(rates[-1])
        losses.append(losses[-1])
        delays.append(delays[-1])
        
    return times, rates, losses, delays

def get_tc_profile_name(folder_path):
    """从文件夹中的log文件名提取tc_profile_name"""
    for f in os.listdir(folder_path):
        if f.endswith('.log'):
            base_name = os.path.splitext(f)[0]
            parts = base_name.split('_')
            if len(parts) > 1: return parts[-1]
    return None

def plot_trace(data_file, save_dir):
    """使用 Plotly 读取data.jsonl并生成五合一的性能图"""
    times, receiving_rates, bandwidth_estimations, packet_loss_ratios, delay_avg_min_diffs, is_heuristic_flags = [], [], [], [], [], []

    def to_numeric(value):
        if isinstance(value, (int, float)): return float(value)
        return 0.0

    with open(data_file, 'r') as f:
        for line in f:
            try:
                d = json.loads(line)
                state = d.get('state', {}); action = d.get('action', {})
                receive_times = state.get('receive_time')
                if not receive_times or not isinstance(receive_times, list): continue
                times.append(receive_times[-1])
                receiving_rates.append(to_numeric(state.get('receiving_rate')))
                bandwidth_estimations.append(to_numeric(action.get('bandwidth_estimation')))
                packet_loss_ratios.append(to_numeric(state.get('packet_loss_ratio')))
                delay_avg_min_diffs.append(to_numeric(state.get('delay_avg_min_diff')))
                # --- 核心改动：从 state 字段提取 isHeuristicUsed ---
                is_heuristic_flags.append(state.get('isHeuristicUsed', False))
            except (json.JSONDecodeError, KeyError, IndexError): continue

    if not times:
        print(f"  No valid data points found in {data_file}"); return

    base_time = times[0]
    times_sec = [(t - base_time) / 1000.0 for t in times]
    total_duration = times_sec[-1] if times_sec else 0

    # --- Plotly 绘图逻辑 ---
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Bandwidth', 'Packet Loss (End-to-End)', 'Queuing Delay (End-to-End)', 'TC Loss (Ground Truth)', 'TC Delay (Ground Truth)')
    )

    # 子图1: 带宽
    fig.add_trace(go.Scatter(x=times_sec, y=np.array(receiving_rates) / 1e6, name='Receiving Rate', mode='lines', line=dict(color='dodgerblue', width=2)), row=1, col=1)
    
    # --- 分段绘制 Bandwidth Estimation ---
    # 将数据按 isHeuristicUsed 分割成连续的段
    segments = []
    if times_sec:
        current_segment = {'times': [], 'values': [], 'is_heuristic': is_heuristic_flags[0]}
        for i in range(len(times_sec)):
            if is_heuristic_flags[i] == current_segment['is_heuristic']:
                current_segment['times'].append(times_sec[i])
                current_segment['values'].append(bandwidth_estimations[i])
            else:
                # 为了线段连续，将当前点也作为上一个线段的终点
                current_segment['times'].append(times_sec[i])
                current_segment['values'].append(bandwidth_estimations[i])
                segments.append(current_segment)
                # 开始新线段
                current_segment = {'times': [times_sec[i]], 'values': [bandwidth_estimations[i]], 'is_heuristic': is_heuristic_flags[i]}
        segments.append(current_segment) # 添加最后一个线段

    # 绘制每个分段
    heuristic_legend_added = False
    non_heuristic_legend_added = False
    for seg in segments:
        if seg['is_heuristic']:
            name = 'BWE (Heuristic)'
            color = 'darkorange'
            show_legend = not heuristic_legend_added
            heuristic_legend_added = True
        else:
            name = 'BWE (Model)'
            color = 'purple'
            show_legend = not non_heuristic_legend_added
            non_heuristic_legend_added = True
        
        fig.add_trace(go.Scatter(
            x=seg['times'], 
            y=np.array(seg['values']) / 1e6, 
            name=name, 
            mode='lines', 
            line=dict(color=color, width=2, dash='dash'),
            legendgroup=name,
            showlegend=show_legend
        ), row=1, col=1)

    # 子图2: 端到端丢包率
    fig.add_trace(go.Scatter(x=times_sec, y=packet_loss_ratios, name='E2E Packet Loss Ratio', mode='lines', line=dict(color='crimson', width=2)), row=2, col=1)

    # 子图3: 端到端延迟
    fig.add_trace(go.Scatter(x=times_sec, y=delay_avg_min_diffs, name='E2E Delay Avg Min Diff', mode='lines', line=dict(color='purple', width=2)), row=3, col=1)

    if TC:
        tc_profile_name = get_tc_profile_name(save_dir)
        if tc_profile_name:
            profile_path = os.path.join(TC_PROFILE_DIR, tc_profile_name)
            commands = parse_tc_profile(profile_path)
            if commands:
                tc_times, tc_rates, tc_losses, tc_delays = get_tc_params_over_time(commands, total_duration)
                if tc_times:
                    truncate_idx = len(tc_times)
                    for i, t in enumerate(tc_times):
                        if t > total_duration:
                            truncate_idx = i
                            break
                    
                    tc_times_truncated = tc_times[:truncate_idx]
                    tc_rates_truncated = tc_rates[:truncate_idx]
                    tc_losses_truncated = tc_losses[:truncate_idx]
                    tc_delays_truncated = tc_delays[:truncate_idx]

                    if tc_times_truncated and tc_times_truncated[-1] < total_duration:
                        tc_times_truncated.append(total_duration)
                        tc_rates_truncated.append(tc_rates_truncated[-1])
                        tc_losses_truncated.append(tc_losses_truncated[-1])
                        tc_delays_truncated.append(tc_delays_truncated[-1])

                    fig.add_trace(go.Scatter(x=tc_times_truncated, y=np.array(tc_rates_truncated) / 1e6, name='TC Rate', mode='lines', line=dict(color='green', width=2.5, dash='dot', shape='hv')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=tc_times_truncated, y=tc_losses_truncated, name='TC Loss', mode='lines', line=dict(color='red', width=2.5, dash='dot', shape='hv')), row=4, col=1)
                    fig.add_trace(go.Scatter(x=tc_times_truncated, y=np.array(tc_delays) * 1000, name='TC Delay', mode='lines', line=dict(color='saddlebrown', width=2.5, dash='dot', shape='hv')), row=5, col=1)

    # 更新图表布局
    fig.update_layout(
        title_text=f"Trace Analysis: {os.path.basename(save_dir)}",
        height=1200,
        legend_traceorder="reversed",
        template="plotly_white"
    )
    fig.update_yaxes(title_text="Rate (Mbps)", autorange=True, row=1, col=1)
    fig.update_yaxes(title_text="Loss Ratio", autorange=True, row=2, col=1)
    fig.update_yaxes(title_text="Delay (ms)", autorange=True, row=3, col=1)
    fig.update_yaxes(title_text="TC Loss Ratio", autorange=True, row=4, col=1)
    fig.update_yaxes(title_text="TC Delay (ms)", autorange=True, row=5, col=1)

    fig.update_yaxes(rangemode="tozero", row=1, col=1)
    fig.update_yaxes(rangemode="tozero", row=3, col=1)
    fig.update_yaxes(rangemode="tozero", row=5, col=1)

    fig.update_xaxes(title_text="Time (s)", row=5, col=1)

    plot_file = os.path.join(save_dir, 'trace_plot.png')
    fig.write_image(plot_file, width=1600, height=1200, scale=1)
    print(f"  Plot saved to {os.path.basename(plot_file)}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot_trace_tc.py <id>")
        sys.exit(1)
    
    test_id = sys.argv[1]
    input_path = os.path.join('results', test_id)

    if not os.path.isdir(input_path):
        print(f"Error: Directory not found at {input_path}")
        sys.exit(1)

    all_folders = sorted([f for f in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, f))])
    
    print(f"--- Starting to process ID: {test_id} ---")
    for i, folder in enumerate(all_folders):
        folder_path = os.path.join(input_path, folder)
        data_file = os.path.join(folder_path, 'data.jsonl')
        plot_file = os.path.join(folder_path, 'trace_plot.png')
        
        print(f"[{i+1}/{len(all_folders)}] Processing folder: {folder}")

        if not os.path.isfile(data_file):
            print("  Skipping: data.jsonl not found.")
            continue
        
        if os.path.isfile(plot_file):
            print("  Skipping: Plot already exists.")
            continue
        
        try:
            plot_trace(data_file, folder_path)
        except Exception as e:
            print(f"  [ERROR] Failed to plot for {folder}: {e}")

    print("--- Done ---")
