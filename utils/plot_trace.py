# input_file是一个jsonl文件，数据每一行是这样的
# {"mi_idx": 5, "state": {"receiving_rate": 420295.5665024631, "num_received_packets": 13, "received_bytes": 10665, "queuing_delay": 41.0, "delay_minus_base": 1758606527988.0, "min_seen_delay": 1758606528147, "delay_ratio": 1.0000000000011373, "delay_avg_min_diff": 2.0, "mean_interarrival": 16.916666666666668, "packet_jitter": 13.493825748590847, "packet_loss_ratio": 0.9980983031012288, "avg_lost_pkts": 3411.5, "video_prob": 1.0, "audio_prob": 0.0, "probe_prob": 0.0, "received_video_bytes": 10333, "received_audio_bytes": 0, "payload_type": [125, 125, 125, 125, 125, 125, 125, 125, 122, 125, 125, 125, 125], "send_time": [46968, 46999, 47035, 47035, 47040, 47066, 47102, 47107, 47112, 47133, 47133, 47164, 47164], "receive_time": [1758606575154, 1758606575186, 1758606575222, 1758606575222, 1758606575227, 1758606575253, 1758606575289, 1758606575294, 1758606575299, 1758606575320, 1758606575328, 1758606575351, 1758606575357]}, "action": {"bandwidth_estimation": 578163.8461538461}}

# 现在我要画图
# 画receiving_rate、bandwidth_estimation、packet_loss_ratio随时间变化
# x轴是时间，y轴是值
# 注意send_time是send端发包的时间
# receive_time是receive端收到包的时间
# send端的时间和receive端的时间不是一个时钟
# 但是send_time和receive_time是一一对应的

# input_path下有好多文件夹，表示一条条trace
# 每个文件夹下都有一个data.jsonl
# input_file就是这个data.jsonl
# input_file就放在data.jsonl同一个路径下

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

input_path = 'results/19'

def get_dynamic_ylim(data, margin=0.05):
    arr = np.array(data)
    if len(arr) == 0:
        return None, None
    lower = np.percentile(arr, 1)
    upper = np.percentile(arr, 99)
    delta = (upper - lower) * margin
    return lower - delta, upper + delta

def plot_trace(data_file, save_dir):
    times = []
    receiving_rates = []
    bandwidth_estimations = []
    packet_loss_ratios = []

    with open(data_file, 'r') as f:
        for line in f:
            d = json.loads(line)
            state = d['state']
            action = d['action']
            t = state['receive_time'][-1] if state['receive_time'] else None
            if t is not None:
                times.append(t)
                receiving_rates.append(state.get('receiving_rate', 0))
                bandwidth_estimations.append(action.get('bandwidth_estimation', 0))
                packet_loss_ratios.append(state.get('packet_loss_ratio', 0))

    if times:
        base_time = times[0]
        times = [(t - base_time) / 1000.0 for t in times]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 左轴：receiving_rate 和 bandwidth_estimation
    y1_min, y1_max = get_dynamic_ylim(receiving_rates + bandwidth_estimations)
    l1, = ax1.plot(times, receiving_rates, color='tab:blue', label='receiving_rate', linewidth=2)
    l2, = ax1.plot(times, bandwidth_estimations, color='tab:orange', label='bandwidth_estimation', linestyle='--', linewidth=2)
    ax1.set_ylabel('Rate (bps)', color='tab:blue', fontsize=14, fontweight='bold')
    if y1_min is not None and y1_max is not None:
        ax1.set_ylim(y1_min, y1_max)
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=12)

    # 右轴：packet_loss_ratio
    ax2 = ax1.twinx()
    y2_min, y2_max = get_dynamic_ylim(packet_loss_ratios)
    l3, = ax2.plot(times, packet_loss_ratios, color='tab:green', label='packet_loss_ratio', linewidth=2)
    ax2.set_ylabel('Packet Loss Ratio', color='tab:green', fontsize=14, fontweight='bold')
    if y2_min is not None and y2_max is not None:
        ax2.set_ylim(max(0, y2_min), min(1, y2_max))
    ax2.tick_params(axis='y', labelcolor='tab:green', labelsize=12)

    # 合并图例
    lines = [l1, l2, l3]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper center', fontsize=13, frameon=True, ncol=3)

    ax1.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    plt.title(os.path.basename(save_dir), fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # 留出标题空间
    plt.savefig(os.path.join(save_dir, 'trace_plot.png'), dpi=150)
    plt.close()

if __name__ == '__main__':
    for folder in os.listdir(input_path):
        folder_path = os.path.join(input_path, folder)
        data_file = os.path.join(folder_path, 'data.jsonl')
        if os.path.isfile(data_file):
            plot_trace(data_file, folder_path)