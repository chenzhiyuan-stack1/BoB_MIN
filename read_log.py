import json

def read_jsonl(filepath):
    """读取jsonl文件，返回每一行的字典组成的列表"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def sum_received_audio_bytes(data):
    """统计所有state.received_audio_bytes的和"""
    total = 0
    for item in data:
        state = item.get('state', {})
        total += state.get('received_audio_bytes', 0)
    return total

def sum_received_video_bytes(data):
    """统计所有state.received_video_bytes的和"""
    total = 0
    for item in data:
        state = item.get('state', {})
        total += state.get('received_video_bytes', 0)
    return total

# 示例用法
if __name__ == "__main__":
    filepath = "results/0/16_06_2025_0924_bob1_1/data.jsonl"
    data = read_jsonl(filepath)
    total_audio_bytes = sum_received_audio_bytes(data)
    total_video_bytes = sum_received_video_bytes(data)
    print(f"received_audio_bytes总和: {total_audio_bytes}")
    print(f"received_video_bytes总和: {total_video_bytes}")