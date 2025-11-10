# basedir下的id文件夹下面有一系列文件夹A、B、C等
# 这一系列文件夹里有data.jsonl文件和log文件
# log文件的名字是这样的webrtc_receive_bob1_bowing_cif_test_30_bad4G.log
# 然后对应的tc_profile文件是bad4G

# data.jsonl文件，数据每一行是这样的，表示一个决策周期下的包信息
# {"mi_idx": 10, "state": {"receiving_rate": 672367.816091954, "num_received_packets": 16, "received_bytes": 14624, "queuing_delay": 0.9375, "delay_minus_base": 1761720599756.9375, "min_seen_delay": 1761720599956, "delay_ratio": 1.0000000000005322, "delay_avg_min_diff": 0.9375, "mean_interarrival": 11.6, "packet_jitter": 11.25547485829515, "packet_loss_ratio": 0, "avg_lost_pkts": 0, "video_prob": 1.0, "audio_prob": 0.0, "probe_prob": 0.0, "received_video_bytes": 14220, "received_audio_bytes": 0, "payload_type": [125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125], "send_time": [12827, 12827, 12827, 12863, 12863, 12863, 12904, 12914, 12925, 12935, 12951, 12961, 12971, 12987, 12997, 13002], "receive_time": [1761720612784, 1761720612784, 1761720612785, 1761720612819, 1761720612820, 1761720612822, 1761720612861, 1761720612871, 1761720612882, 1761720612892, 1761720612907, 1761720612918, 1761720612929, 1761720612943, 1761720612953, 1761720612958], "sequence_number": [11231, 11232, 11233, 11234, 11235, 11236, 11237, 11238, 11239, 11240, 11241, 11242, 11243, 11244, 11245, 11246], "all_payload_type": [125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125], "all_sequence_number": [11231, 11232, 11233, 11234, 11235, 11236, 11237, 11238, 11239, 11240, 11241, 11242, 11243, 11244, 11245, 11246], "all_send_timestamp": [12827, 12827, 12827, 12863, 12863, 12863, 12904, 12914, 12925, 12935, 12951, 12961, 12971, 12987, 12997, 13002], "all_ssrc": [2803628529, 2803628529, 2803628529, 2803628529, 2803628529, 2803628529, 2803628529, 2803628529, 2803628529, 2803628529, 2803628529, 2803628529, 2803628529, 2803628529, 2803628529, 2803628529], "all_padding_length": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "all_header_length": [24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 44, 24, 24], "all_receive_timestamp": [1761720612784, 1761720612784, 1761720612785, 1761720612819, 1761720612820, 1761720612822, 1761720612861, 1761720612871, 1761720612882, 1761720612892, 1761720612907, 1761720612918, 1761720612929, 1761720612943, 1761720612953, 1761720612958], "all_payload_size": [879, 880, 880, 869, 870, 870, 1033, 1033, 1033, 1033, 1034, 1034, 1034, 1014, 333, 391], "all_bandwidth_prediction": [1175139, 1175139, 1175139, 1175139, 1175139, 1175139, 1175139, 1175139, 1175139, 1175139, 1175139, 1175139, 1175139, 1175139, 1175139, 1175139]}, "action": {"bandwidth_estimation": 1707388}}
# 这个决策周期的state主要关注这几个字段：
# "receiving_rate": 290262.62626262626,
# "num_received_packets": 11,
# "received_bytes": 7184,
# "queuing_delay": 100.636474609375,
# "delay_minus_base": 1761580697322.6365,
# "min_seen_delay": 1761580697422,
# "delay_ratio": 1.0000000000003613,
# "delay_avg_min_diff": 0.636474609375,
# "mean_interarrival": 19.8,
# "packet_jitter": 13.9427400463467,
# "packet_loss_ratio": 0.08333333333333333,
# 这个时刻的state_t = [receiving_rate, num_received_packets, received_bytes, queuing_delay, delay_minus_base, min_seen_delay, delay_ratio, delay_avg_min_diff, mean_interarrival, packet_jitter, packet_loss_ratio]
# 如果不足前5个时刻的数据，就用补0的方式补齐
# 然后真实的state = cat([state_t-5, state_t-4, state_t-3, state_t-2, state_t-1, state_t]) 是一个66维的向量，1-11是state_t-5，12-22是state_t-4，依次类推，最后11个是state_t
# next_state同理，是下一个时刻真实的state，如果没有下一个时刻了，就补0向量
# observation就是这个真实的state
# next_observation就是下一个时刻的真实state

# action = bandwidth_estimation
# reward = -(queuing_delay / 100 + 5 * packet_loss_ratio) + receiving_rate / 1000000，希望每一项都归一化到0-1之间

# 如果这是data.jsonl文件的最后一个决策周期，那么terminal = 1，否则0
# 注意data.jsonl文件最后一行是空行，不要读进去

# 如果TC为True的话，还要记录tc限制真实带宽、tc加的丢包率、tc加的延迟随时间的变化（tc加的丢包率和延迟初始值为0）
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

# true_capacity是当前决策周期对应的tc限制的真实带宽，单位bps
# true_loss_rate是当前决策周期对应的tc加的丢包率，转换为小数
# true_delay是当前决策周期对应的tc加的延迟，单位ms

# 然后tc相关的信息要和data.jsonl文件里的决策周期一一对应起来
# data.jsonl文件里的
# send_time是send端发包的时间
# receive_time是receive端收到包的时间
# send端的时间和receive端的时间不是一个时钟
# 但是send_time和receive_time是一一对应的，可以用send_time（平均时间对应决策周期的时间）来推断当前决策周期对应的tc_profile配置
# 可以参考plot_trace_tc.py脚本

# 写个函数，输入basedir和ids（id的列表），输出一个字典
# 把整个数据集的结构转换成下面的形式
# data = {}
# data['observations']: (6534914, 66)
# data['actions']: (6534914, 1)
# data['next_observations']: (6534914, 66)
# data['rewards']: (6534914,)
# data['terminals']: (6534914,)
# data['true_capacities']: (6534914,)
# data['true_loss_rates']: (6534914,)
# data['true_delays']: (6534914,)

# 再写个函数，把上面的字典保存成pickle文件

import os
import json
import numpy as np
import pickle
from collections import deque

# --- TC Profile 解析函数 (保持不变) ---
TC_PROFILE_DIR = '/home/min414/data2/BoB_MIN/tc_profiles'

def parse_unit(value_str):
    value_str = str(value_str).lower().strip()
    if 'kbit' in value_str: return float(value_str.replace('kbit', '')) * 1000
    if 'mbit' in value_str: return float(value_str.replace('mbit', '')) * 1000000
    if 'ms' in value_str: return float(value_str.replace('ms', ''))
    if 's' in value_str: return float(value_str.replace('s', '')) * 1000
    if '%' in value_str: return float(value_str.replace('%', '')) / 100.0
    try: return float(value_str)
    except (ValueError, TypeError): return 0.0

def parse_tc_profile(profile_path):
    if not os.path.exists(profile_path): return None
    with open(profile_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    if not lines: return []
    
    rate_groups = []
    current_group = []
    for line in lines:
        if line.startswith('rate'):
            if current_group: rate_groups.append(current_group)
            current_group = [line]
        else: current_group.append(line)
    if current_group: rate_groups.append(current_group)

    commands = []
    for group in rate_groups:
        params = {'rate': 0, 'loss': 0, 'delay': 0, 'duration': 0}
        for cmd in group:
            parts = cmd.split()
            if len(parts) < 2: continue
            cmd_type, cmd_value = parts[0], parts[1]
            if cmd_type == 'rate': params['rate'] = parse_unit(cmd_value)
            elif cmd_type == 'loss': params['loss'] = parse_unit(cmd_value)
            elif cmd_type == 'delay': params['delay'] = parse_unit(cmd_value)
            elif cmd_type == 'wait': params['duration'] += parse_unit(cmd_value)
        commands.append(params)
    return commands

def get_tc_params_at_time(commands, time_ms):
    if not commands: return 0, 0, 0
    total_cycle_duration = sum(cmd['duration'] for cmd in commands)
    if total_cycle_duration <= 0:
        cmd = commands[0]
        return cmd['rate'], cmd['loss'], cmd['delay']
    time_in_cycle = time_ms % total_cycle_duration
    elapsed_time = 0
    for cmd in commands:
        if elapsed_time + cmd['duration'] > time_in_cycle:
            return cmd['rate'], cmd['loss'], cmd['delay']
        elapsed_time += cmd['duration']
    last_cmd = commands[-1]
    return last_cmd['rate'], last_cmd['loss'], last_cmd['delay']

def get_tc_profile_name(folder_path):
    for f in os.listdir(folder_path):
        if f.endswith('.log'):
            base_name = os.path.splitext(f)[0]
            parts = base_name.split('_')
            if len(parts) > 1: return parts[-1]
    return None

# --- 新增：可复用的状态构建函数 ---
def update_and_get_observation(history_deque: deque, new_state_t: np.ndarray) -> np.ndarray:
    """
    更新状态历史队列并返回拼接好的、平坦化的 observation 向量。
    
    :param history_deque: 存储历史状态的 deque 对象。
    :param new_state_t: 当前时间步的 11 维状态向量。
    :return: 拼接好的 66 维 observation 向量。
    """
    history_deque.append(new_state_t)
    observation = np.concatenate(list(history_deque)).flatten()
    return observation

# --- 数据集构建核心函数 ---
def process_results_to_dataset(basedir, ids, state_window_size=6):
    from tqdm import tqdm
    dataset = {
        'observations': [], 'actions': [], 'next_observations': [],
        'rewards': [], 'terminals': [], 'true_capacities': [],
        'true_loss_rates': [], 'true_delays': []
    }
    
    state_keys = [
        "receiving_rate", "num_received_packets", "received_bytes", "queuing_delay",
        "delay_minus_base", "min_seen_delay", "delay_ratio", "delay_avg_min_diff",
        "mean_interarrival", "packet_jitter", "packet_loss_ratio"
    ]
    state_dim = len(state_keys)
    zero_state_t = np.array([0.0] * state_dim)

    for id_val in tqdm(ids, desc="Processing IDs"):
        id_path = os.path.join(basedir, str(id_val))
        if not os.path.isdir(id_path):
            tqdm.write(f"Warning: Directory not found for id {id_val}, skipping.")
            continue

        trace_folders = sorted([f for f in os.listdir(id_path) if os.path.isdir(os.path.join(id_path, f))])
        
        for folder in tqdm(trace_folders, desc=f"  ID {id_val} Traces", leave=False):
            folder_path = os.path.join(id_path, folder)
            data_file = os.path.join(folder_path, 'data.jsonl')
            if not os.path.exists(data_file): continue

            tc_profile_name = get_tc_profile_name(folder_path)
            tc_commands = parse_tc_profile(os.path.join(TC_PROFILE_DIR, tc_profile_name)) if tc_profile_name else None

            with open(data_file, 'r') as f: lines = f.read().splitlines()
            if not lines: continue

            trace_data = [json.loads(line) for line in lines]
            num_steps = len(trace_data)
            
            trace_states_t = [np.array([float(step.get('state', {}).get(k, 0.0)) for k in state_keys]) for step in trace_data]

            state_history = deque([zero_state_t] * state_window_size, maxlen=state_window_size)

            for i in range(num_steps):
                # --- 使用新的复用函数构建 observation ---
                observation = update_and_get_observation(state_history, trace_states_t[i])

                if i + 1 < num_steps:
                    # 构建 next_observation
                    next_state_history = deque(state_history, maxlen=state_window_size) # 复制当前历史
                    next_observation = update_and_get_observation(next_state_history, trace_states_t[i+1])
                else:
                    next_observation = np.zeros_like(observation)

                action = trace_data[i].get('action', {}).get('bandwidth_estimation', 0.0)
                
                state_dict = trace_data[i].get('state', {})
                reward = -(state_dict.get('queuing_delay', 0.0) / 100.0 + 5.0 * state_dict.get('packet_loss_ratio', 0.0)) + (state_dict.get('receiving_rate', 0.0) / 1000000.0)
                terminal = 1 if (i == num_steps - 1) else 0

                true_capacity, true_loss, true_delay = 0, 0, 0
                if tc_commands and state_dict.get('send_time'):
                    avg_send_time_ms = np.mean(state_dict['send_time'])
                    true_capacity, true_loss, true_delay = get_tc_params_at_time(tc_commands, avg_send_time_ms)

                dataset['observations'].append(observation)
                dataset['actions'].append([action])
                dataset['next_observations'].append(next_observation)
                dataset['rewards'].append(reward)
                dataset['terminals'].append(terminal)
                dataset['true_capacities'].append(true_capacity)
                dataset['true_loss_rates'].append(true_loss)
                dataset['true_delays'].append(true_delay)

    print("\nConverting lists to numpy arrays...")
    for key, value in dataset.items():
        dataset[key] = np.array(value)
        print(f"Final shape for '{key}': {dataset[key].shape}")
    return dataset

def save_dataset_as_pickle(dataset, output_path):
    print(f"\nSaving dataset to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    print("Save complete.")

if __name__ == '__main__':
    basedir = '/home/min414/data2/extra_storage'
    ids = ['5']
    output_filename = '/home/min414/data2/extra_storage/BoB_5.pickle'
    final_dataset = process_results_to_dataset(basedir, ids)
    if final_dataset['observations'].shape[0] > 0:
        save_dataset_as_pickle(final_dataset, output_filename)
    else:
        print("No data was processed. Pickle file not created.")
