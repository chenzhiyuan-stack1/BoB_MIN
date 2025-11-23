import os
import re
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 配置区域 ---

# 日志文件路径
LOG_FILE_PATH = '/home/min414/data2/BoB_MIN/results_exp/251119105623_online_exp_online_8cd7/debug.log'

# 定义要绘制的指标分组
PLOTS_CONFIG = [
    # 第1行：训练 Loss (通常量级差异很大，双轴很有用)
    ("Training Losses", [
        'Train/ql_loss', 
        'Train/actor_loss', 
        'Train/v_loss',
        'Train/bc_loss'
    ]),
    
    # 第2行：Q值与V值 (这些通常量级一致，应该会都在左轴)
    ("Value Estimates", [
        'Train/v', 
        'Train/q1', 
        'Train/q2', 
        'Train/target_q',
        'Train/next_v',
    ]),
    
    # 第3行：优势函数统计
    ("Advantage Statistics", [
        'Train/adv_mean', 
        'Train/adv_std'
    ]),
    
    # 第4行：评估指标
    ("Evaluation Metrics", [
        'Eval/Accuracy', 
        'Eval/MSE',
        'Eval/Overestimation'
    ]),
    
    # 第5行：特定 Trace 的表现
    ("Trace Performance (WIFI)", [
        'Trace/MDQL_WIFI/avg_receiving_rate',
        'Trace/MDQL_WIFI/avg_delay',
        'Trace/MDQL_WIFI/avg_loss'
    ])
]

# --- 脚本逻辑 ---

def parse_log_file(file_path):
    """解析日志文件，返回 DataFrame"""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    data = []
    # 正则表达式匹配：Timestamp | [ID] MetricName Value
    pattern = re.compile(r'\|\[.*?\]\s+(.*?)\s+([-\d\.eE]+)')
    step_pattern = re.compile(r'\|\[.*?\]\s+Online/Step\s+(\d+)')
    
    current_step = -1
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # 尝试匹配 Step
            step_match = step_pattern.search(line)
            if step_match:
                current_step = int(step_match.group(1))
                continue
            
            # 尝试匹配指标
            match = pattern.search(line)
            if match and current_step >= 0:
                metric_name = match.group(1).strip()
                try:
                    metric_value = float(match.group(2))
                    data.append({
                        'Step': current_step,
                        'Metric': metric_name,
                        'Value': metric_value
                    })
                except ValueError:
                    continue

    if not data:
        print("No valid data found in log file.")
        return None
        
    df = pd.DataFrame(data)
    return df

def plot_metrics(df, config, output_file):
    """使用 Plotly 绘制图表并保存为 PNG，支持双Y轴自适应"""
    rows = len(config)
    
    # 启用 secondary_y (双Y轴)
    # specs 必须是一个 list of lists，对应网格的每一行
    specs = [[{"secondary_y": True}] for _ in range(rows)]
    
    fig = make_subplots(
        rows=rows, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[item[0] for item in config],
        specs=specs
    )

    # 颜色池
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    for i, (title, metrics) in enumerate(config):
        row_idx = i + 1
        
        # 用于判断是否需要使用右轴的基准值
        base_mean = None
        
        for j, metric_name in enumerate(metrics):
            # 筛选该指标的数据
            metric_data = df[df['Metric'] == metric_name]
            
            if metric_data.empty:
                print(f"Warning: Metric '{metric_name}' not found in log data.")
                continue
            
            # 按 Step 排序
            metric_data = metric_data.sort_values('Step')
            current_mean = metric_data['Value'].abs().mean()
            
            # --- 自动判断 Y 轴逻辑 ---
            use_secondary_y = False
            axis_label = "(L)"
            
            if j == 0:
                # 第一个指标总是作为左轴基准
                base_mean = current_mean if current_mean != 0 else 1.0
            else:
                # 计算当前指标与基准指标的比率
                ratio = current_mean / base_mean if base_mean != 0 else 0
                
                # 如果差异超过 5 倍 或 小于 0.2 倍，则放入右轴
                # 这样可以保证量级差异大的指标也能看清趋势
                if ratio > 5.0 or ratio < 0.2:
                    use_secondary_y = True
                    axis_label = "(R)"
            
            # 添加 Trace
            fig.add_trace(
                go.Scatter(
                    x=metric_data['Step'],
                    y=metric_data['Value'],
                    name=f"{metric_name} {axis_label}", # 在图例中标记轴
                    mode='lines+markers',
                    line=dict(width=2, color=colors[j % len(colors)]),
                    marker=dict(size=3),
                    legendgroup=str(row_idx),
                    legendgrouptitle_text=title
                ),
                row=row_idx,
                col=1,
                secondary_y=use_secondary_y
            )
            
            # 设置Y轴标题（可选，这里为了简洁只在图例区分）
            # if use_secondary_y:
            #     fig.update_yaxes(title_text="Secondary", row=row_idx, col=1, secondary_y=True)

    # 更新布局
    fig.update_layout(
        title_text=f"Training Metrics Analysis: {os.path.basename(LOG_FILE_PATH)}",
        height=400 * rows,  # 稍微增加高度
        width=1600,
        template="plotly_white",
        showlegend=True,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(groupclick="toggleitem") # 点击图例组标题可以切换整组
    )
    
    # 给最底部的x轴加标签
    fig.update_xaxes(title_text="Online Step", row=rows, col=1)

    # 保存为 PNG
    try:
        fig.write_image(output_file, scale=2)
        print(f"Plot saved to {output_file}")
    except ValueError as e:
        print(f"Error saving image: {e}")
        print("Make sure you have installed kaleido: pip install -U kaleido")

if __name__ == "__main__":
    # 如果命令行传入了文件路径，则使用命令行的，否则使用脚本开头的配置
    log_path = sys.argv[1] if len(sys.argv) > 1 else LOG_FILE_PATH
    
    print(f"Processing log file: {log_path}")
    df = parse_log_file(log_path)
    
    if df is not None:
        output_png = os.path.join(os.path.dirname(log_path), 'training_metrics.png')
        plot_metrics(df, PLOTS_CONFIG, output_png)