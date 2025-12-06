import os
import json
import numpy as np
import sys
import re

# 导入绘图库 (保持使用 Plotly)
import plotly.graph_objects as go

# --- 全局配置 ---
TC = True
TC_PROFILE_DIR = '/home/min414/data2/BoB_MIN/tc_profiles'

def parse_unit(value_str):
    """解析带有单位的字符串"""
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
    """解析tc profile文件"""
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
        params = {'rate': 0, 'loss': 0, 'delay': 0, 'duration': 0}
        for cmd in group:
            parts = cmd.split()
            if len(parts) < 2: continue
            cmd_type = parts[0]
            cmd_value = parts[1]
            if cmd_type == 'rate': params['rate'] = parse_unit(cmd_value)
            elif cmd_type == 'loss': params['loss'] = parse_unit(cmd_value)
            elif cmd_type == 'delay': params['delay'] = parse_unit(cmd_value)
            elif cmd_type == 'wait': params['duration'] += parse_unit(cmd_value)
        commands.append(params)
    return commands

def get_tc_params_over_time(commands, total_duration):
    """生成循环的TC参数序列"""
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
    """提取tc_profile_name"""
    for f in os.listdir(folder_path):
        if f.endswith('.log'):
            base_name = os.path.splitext(f)[0]
            parts = base_name.split('_')
            if len(parts) > 1: return parts[-1]
    return None

def plot_trace(data_file, save_dir):
    """读取data.jsonl并生成单张高样式的带宽对比图"""
    times, receiving_rates, bandwidth_estimations, is_heuristic_flags = [], [], [], []

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
                # 提取 isHeuristicUsed
                is_heuristic_flags.append(state.get('isHeuristicUsed', False))
            except (json.JSONDecodeError, KeyError, IndexError): continue

    if not times:
        print(f"  No valid data points found in {data_file}"); return

    base_time = times[0]
    times_sec = [(t - base_time) / 1000.0 for t in times]
    total_duration = times_sec[-1] if times_sec else 0

    # --- 创建单张图表 ---
    fig = go.Figure()

    # 1. 绘制 True Bandwidth (TC Rate) - 黑色实线，最先绘制作为参考
    if TC:
        tc_profile_name = get_tc_profile_name(save_dir)
        if tc_profile_name:
            profile_path = os.path.join(TC_PROFILE_DIR, tc_profile_name)
            commands = parse_tc_profile(profile_path)
            if commands:
                tc_times, tc_rates, _, _ = get_tc_params_over_time(commands, total_duration)
                if tc_times:
                    # 截断逻辑
                    truncate_idx = len(tc_times)
                    for i, t in enumerate(tc_times):
                        if t > total_duration:
                            truncate_idx = i
                            break
                    tc_times_truncated = tc_times[:truncate_idx]
                    tc_rates_truncated = tc_rates[:truncate_idx]
                    if tc_times_truncated and tc_times_truncated[-1] < total_duration:
                        tc_times_truncated.append(total_duration)
                        tc_rates_truncated.append(tc_rates_truncated[-1])

                    # 黑色粗线表示真实带宽
                    fig.add_trace(go.Scatter(
                        x=tc_times_truncated, 
                        y=np.array(tc_rates_truncated) / 1e6, 
                        name='True Bandwidth', 
                        mode='lines', 
                        line=dict(color='black', width=4)
                    ))

    # 2. 绘制 Receiving Rate - 绿色细线 (类似 Behavior Policy)
    fig.add_trace(go.Scatter(
        x=times_sec, 
        y=np.array(receiving_rates) / 1e6, 
        name='Receiving Rate', 
        mode='lines', 
        line=dict(color='green', width=3)
    ))

    # 3. 绘制 Bandwidth Estimation (Policy) - 分段绘制
    segments = []
    if times_sec:
        current_segment = {'times': [], 'values': [], 'is_heuristic': is_heuristic_flags[0]}
        for i in range(len(times_sec)):
            if is_heuristic_flags[i] == current_segment['is_heuristic']:
                current_segment['times'].append(times_sec[i])
                current_segment['values'].append(bandwidth_estimations[i])
            else:
                current_segment['times'].append(times_sec[i])
                current_segment['values'].append(bandwidth_estimations[i])
                segments.append(current_segment)
                current_segment = {'times': [times_sec[i]], 'values': [bandwidth_estimations[i]], 'is_heuristic': is_heuristic_flags[i]}
        segments.append(current_segment)

    heuristic_legend_added = False
    non_heuristic_legend_added = False
    
    for seg in segments:
        if seg['is_heuristic']:
            # 启发式/规则 -> 蓝色
            name = 'BWE (Heuristic)'
            color = 'blue'
            show_legend = not heuristic_legend_added
            heuristic_legend_added = True
        else:
            # 模型/策略 -> 红色 (类似参考图中的 Meta Diffusion Policy，重点突出)
            name = 'BWE (Model Policy)'
            color = 'red'
            show_legend = not non_heuristic_legend_added
            non_heuristic_legend_added = True
        
        fig.add_trace(go.Scatter(
            x=seg['times'], 
            y=np.array(seg['values']) / 1e6, 
            name=name, 
            mode='lines', 
            line=dict(color=color, width=3), # 实线更清晰
            legendgroup=name,
            showlegend=show_legend
        ))

    # --- 样式调整 (Big Text & Paper Style) ---
    fig.update_layout(
        # 标题和背景
        title_text="", # 去掉顶部标题，更像论文插图，如果需要可以加回去
        template="plotly_white",
        autosize=False,
        width=1200, # 宽一点
        height=700,
        
        # 字体设置 (Big Text)
        font=dict(
            family="Times New Roman", # 学术常用字体
            size=26, # 全局大字体
            color="black"
        ),
        
        # 图例设置 (放在内部右下角或合适位置)
        legend=dict(
            x=0.98,
            y=0.05,
            xanchor="right",
            yanchor="bottom",
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="Black",
            borderwidth=1,
            font=dict(size=24)
        ),
        
        # 边距
        margin=dict(l=80, r=40, t=40, b=80),
    )

    # X轴设置
    fig.update_xaxes(
        title_text="Call Duration (second)",
        showline=True, linewidth=2, linecolor='black', mirror=True,
        tickfont=dict(size=24),
        title_font=dict(size=28)
    )

    # Y轴设置
    fig.update_yaxes(
        title_text="Bandwidth (Mbps)",
        showline=True, linewidth=2, linecolor='black', mirror=True,
        tickfont=dict(size=24),
        title_font=dict(size=28),
        rangemode="tozero" # 从0开始
    )

    # 保存图片
    plot_file = os.path.join(save_dir, 'trace_plot_bandwidth.png')
    fig.write_image(plot_file, width=1200, height=700, scale=2) # scale=2 保证高清
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
        # 注意：如果不想覆盖原图，可以改文件名，这里我改成了 trace_plot_bandwidth.png
        
        print(f"[{i+1}/{len(all_folders)}] Processing folder: {folder}")

        if not os.path.isfile(data_file):
            print("  Skipping: data.jsonl not found.")
            continue
        
        try:
            plot_trace(data_file, folder_path)
        except Exception as e:
            print(f"  [ERROR] Failed to plot for {folder}: {e}")

    print("--- Done ---")