import json

def sum_received_bytes(filepath):
    total = 0
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                state = item.get('state', {})
                total += state.get('received_bytes', 0)
    return total

def sum_received_packets(filepath):
    count = 0
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                state = item.get('state', {})
                count += state.get('num_received_packets', 0)
    return count

# 示例用法
filepath = "results/4/06_09_2025_1705_heuristic2_2/data.jsonl"
print("总收到的包的大小:", sum_received_bytes(filepath), "bytes")
print("总收到的包的数量:", sum_received_packets(filepath), "packets")