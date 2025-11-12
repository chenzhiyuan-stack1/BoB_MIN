import os
import json
import numpy as np
from tqdm import tqdm
# --- 核心改动：导入 Plotly 库 ---
import plotly.graph_objects as go

def analyze_metrics(basedir, ids):
    """
    遍历所有指定的数据文件，提取关键指标并进行统计分析。
    """
    all_queuing_delays = []
    all_packet_loss_ratios = []
    all_receiving_rates = []

    print("Starting data extraction...")
    for id_val in tqdm(ids, desc="Processing IDs"):
        id_path = os.path.join(basedir, str(id_val))
        if not os.path.isdir(id_path):
            tqdm.write(f"Warning: Directory not found for id {id_val}, skipping.")
            continue

        trace_folders = sorted([f for f in os.listdir(id_path) if os.path.isdir(os.path.join(id_path, f))])
        
        for folder in tqdm(trace_folders, desc=f"  ID {id_val} Traces", leave=False):
            folder_path = os.path.join(id_path, folder)
            data_file = os.path.join(folder_path, 'data.jsonl')
            if not os.path.exists(data_file):
                continue

            with open(data_file, 'r') as f:
                lines = f.read().splitlines()

            for line in lines:
                if not line: continue
                try:
                    d = json.loads(line)
                    state = d.get('state', {})
                    
                    # 提取并验证指标
                    queuing_delay = state.get('queuing_delay')
                    if queuing_delay is not None and np.isfinite(queuing_delay):
                        all_queuing_delays.append(queuing_delay)

                    packet_loss_ratio = state.get('packet_loss_ratio')
                    if packet_loss_ratio is not None and np.isfinite(packet_loss_ratio):
                        all_packet_loss_ratios.append(packet_loss_ratio)

                    receiving_rate = state.get('receiving_rate')
                    if receiving_rate is not None and np.isfinite(receiving_rate):
                        all_receiving_rates.append(receiving_rate)

                except json.JSONDecodeError:
                    continue
    
    print("\nData extraction complete. Performing analysis...")
    
    # --- 核心分析函数 ---
    def print_stats(name, data):
        if not data:
            print(f"\n--- Analysis for {name} ---")
            print("No data found.")
            return

        arr = np.array(data)
        print(f"\n--- Analysis for {name} ---")
        print(f"Total data points: {len(arr)}")
        print(f"Mean: {np.mean(arr):.4f}")
        print(f"Std Dev: {np.std(arr):.4f}")
        print(f"Min: {np.min(arr):.4f}")
        print(f"Max: {np.max(arr):.4f}")
        
        percentiles = [50, 75, 90, 95, 99, 99.9]
        p_values = np.percentile(arr, percentiles)
        for p, v in zip(percentiles, p_values):
            print(f"{p}th percentile: {v:.4f}")
            
        # --- 核心改动：使用 Plotly 绘制直方图 ---
        fig = go.Figure()
        
        # 添加直方图轨迹
        fig.add_trace(go.Histogram(
            x=arr,
            name=name,
            nbinsx=100  # 设置柱子的数量
        ))

        # 更新图表布局
        fig.update_layout(
            title_text=f'Distribution of {name}',
            xaxis_title_text='Value',
            yaxis_title_text='Frequency',
            bargap=0.1, # 柱子之间的间隙
            # 使用99.9分位数作为范围，以更好地可视化主体分布
            xaxis_range=[np.min(arr), np.percentile(arr, 99.9)]
        )
        
        # 保存为可交互的 HTML 文件
        html_filename = f'{name.replace(" ", "_").lower()}_distribution.html'
        fig.write_html(html_filename)
        print(f"Interactive histogram saved to {html_filename}")


    print_stats("Queuing Delay (ms)", all_queuing_delays)
    print_stats("Packet Loss Ratio", all_packet_loss_ratios)
    print_stats("Receiving Rate (bps)", all_receiving_rates)


if __name__ == '__main__':
    # --- 配置区 ---
    # 请根据您的数据存放位置修改
    basedir = '/home/min414/data2/extra_storage'
    # 选择您想要分析的数据集ID
    ids = ['0', '1', '2', '3', '4', '5', '6', '7'] 
    # --- 配置结束 ---
    
    analyze_metrics(basedir, ids)
