import math

# 假设payload_type定义如下（请根据实际协议调整）
VIDEO_TYPES = {96,97,98,99,100,101,127,123,125,122,124}
AUDIO_TYPES = {111,103,104,9,102,0,8,106,105,13,110,112,113,126}
PROBE_TYPES = {100}
BASE_DELAY = 200  # ms

# 1. Receiving rate (bps)
# Receiving rate: rate at which the client receives data from the sender during a MI, unit: bps.
def receiving_rate(packets_list):
    if not packets_list or len(packets_list) < 2:
        return 0
    total_bytes = sum(pkt.size for pkt in packets_list)
    duration = packets_list[-1].receive_timestamp - packets_list[0].receive_timestamp
    if duration <= 0:
        return 0
    return total_bytes * 8 * 1000 / duration

# 2. Number of received packets
# Number of received packets: total number of packets received in a MI, unit: packet.
def num_received_packets(packets_list):
    return len(packets_list)

# 3. Received bytes
# Received bytes: total number of bytes received in a MI, unit: Bytes.
def received_bytes(packets_list):
    return sum(pkt.size for pkt in packets_list)

# 4. Queuing delay
# Queuing delay: average delay of packets received in a MI minus the minimum packet delay observed so far, unit: ms.
def queuing_delay(packets_list, min_seen_delay):
    if not packets_list:
        return 0
    delays = [pkt.receive_timestamp - pkt.send_timestamp for pkt in packets_list]
    avg_delay = sum(delays) / len(delays)
    return avg_delay - min_seen_delay if min_seen_delay is not None else 0

# 5. Delay (avg delay - base delay) 没啥用
# Delay: average delay of packets received in a MI minus a fixed base delay of 200ms, unit: ms.
def delay_minus_base(packets_list, base_delay=BASE_DELAY):
    if not packets_list:
        return 0
    delays = [pkt.receive_timestamp - pkt.send_timestamp for pkt in packets_list]
    avg_delay = sum(delays) / len(delays)
    return avg_delay - base_delay

# 6. Minimum seen delay (全局最小)
# Minimum seen delay: minimum packet delay observed so far, unit: ms.
def min_seen_delay(packets_list, prev_min=None):
    if not packets_list:
        return prev_min if prev_min is not None else 0
    delays = [pkt.receive_timestamp - pkt.send_timestamp for pkt in packets_list]
    min_delay = min(delays)
    if prev_min is not None:
        return min(min_delay, prev_min)
    return min_delay

# 7. Delay ratio (avg delay / min delay in MI)
# Delay ratio: average delay of packets received in a MI divided by the minimum delay of packets received in the same MI, unit: ms/ms.
def delay_ratio(packets_list):
    if not packets_list:
        return 0
    delays = [pkt.receive_timestamp - pkt.send_timestamp for pkt in packets_list]
    avg_delay = sum(delays) / len(delays)
    min_delay = min(delays)
    return avg_delay / min_delay if min_delay > 0 else float('inf')

# 8. Delay average minimum difference
# Delay average minimum difference: average delay of packets received in a MI minus the minimum delay of packets received in the same MI, unit: ms.
def delay_avg_min_diff(packets_list):
    if not packets_list:
        return 0
    delays = [pkt.receive_timestamp - pkt.send_timestamp for pkt in packets_list]
    avg_delay = sum(delays) / len(delays)
    min_delay = min(delays)
    return avg_delay - min_delay

# 9. Packet interarrival time (mean)
# Packet interarrival time: mean interarrival time of packets received in a MI, unit: ms.
def mean_interarrival(packets_list):
    if len(packets_list) < 2:
        return 0
    arrival_times = [pkt.receive_timestamp for pkt in packets_list]
    interarrivals = [arrival_times[i] - arrival_times[i-1] for i in range(1, len(arrival_times))]
    return sum(interarrivals) / len(interarrivals)

# 10. Packet jitter (stddev of interarrival)
# Packet jitter: standard deviation of interarrival time of packets received in a MI, unit: ms.
def packet_jitter(packets_list):
    """Packet jitter: standard deviation of interarrival time of packets received in a MI, unit: ms."""
    if len(packets_list) < 2:
        return 0
    arrival_times = [pkt.receive_timestamp for pkt in packets_list]
    interarrivals = [arrival_times[i] - arrival_times[i-1] for i in range(1, len(arrival_times))]
    mean_ia = sum(interarrivals) / len(interarrivals)
    if len(interarrivals) < 2:
        return 0
    variance = sum((x - mean_ia) ** 2 for x in interarrivals) / (len(interarrivals) - 1)
    return math.sqrt(variance)

# 11. Packet loss ratio
# Packet loss ratio: probability of packet loss in a MI, unit: packet/packet.
# def packet_loss_ratio(packets_list):
#     seqs = [pkt.sequence_number for pkt in packets_list]
#     if not seqs:
#         return 0
#     expected = max(seqs) - min(seqs) + 1
#     received = len(seqs)
#     return 1 - received / expected if expected > 0 else 0
def packet_loss_ratio(packets_list):
    # 【关键改动】: 过滤掉重传包 (RTX, type 122) 和其他非视频媒体包。
    # 丢包率必须在单一的、原始的媒体流上计算，否则序列号空间不同会导致计算结果完全错误。
    # 我们只关心原始视频包 (如 type 125) 的序列号。
    seqs = sorted([pkt.sequence_number for pkt in packets_list if pkt.payload_type not in {122, 124}]) # 122是RTX, 124是ULPFEC

    if len(seqs) < 2: # 如果过滤后包太少，无法判断丢包，则认为没有丢包
        return 0
    
    # 使用过滤后的序列号进行计算
    expected = max(seqs) - min(seqs) + 1
    received = len(seqs)

    # 检查序列号是否发生回绕 (wrap-around)，这是一个健壮性处理
    # WebRTC 的序列号是 16-bit 的，会从 65535 回绕到 0
    if expected > 30000: # 如果序列号跨度异常大，很可能是发生了回绕
        # 简单的回绕处理：找到最大的间隔，认为那里是回绕点
        max_gap = 0
        for i in range(1, len(seqs)):
            gap = seqs[i] - seqs[i-1]
            if gap > max_gap:
                max_gap = gap
        
        # 从总跨度中减去这个巨大的“伪”间隔
        expected = expected - max_gap + 1

    if expected <= 0:
        return 0

    loss_ratio = (expected - received) / expected
    return max(0, loss_ratio) # 确保结果不会是负数

# 12. Average number of lost packets (每次丢包的平均丢包数)
# Average number of lost packets: average number of lost packets given a loss occurs, unit: packet.
# def avg_lost_pkts(packets_list):
#     seqs = sorted(pkt.sequence_number for pkt in packets_list)
#     lost_counts = []
#     for i in range(1, len(seqs)):
#         gap = seqs[i] - seqs[i-1] - 1
#         if gap > 0:
#             lost_counts.append(gap)
#     return sum(lost_counts) / len(lost_counts) if lost_counts else 0
def avg_lost_pkts(packets_list):
    # 【关键改动】: 同样，只在原始媒体流上计算丢包间隔。
    seqs = sorted([pkt.sequence_number for pkt in packets_list if pkt.payload_type not in {122, 124}])

    if len(seqs) < 2:
        return 0

    lost_counts = []
    for i in range(1, len(seqs)):
        gap = seqs[i] - seqs[i-1] - 1
        # 同样需要考虑序列号回绕，忽略异常大的 gap
        if gap > 0 and gap < 1000: # 假设一次连续丢包不会超过1000个
            lost_counts.append(gap)
    return sum(lost_counts) / len(lost_counts) if lost_counts else 0

# 13. Video packets probability
# Video packets probability: proportion of video packets in the packets received in a MI, unit: packet/packet.
def video_prob(packets_list):
    if not packets_list:
        return 0
    video_cnt = sum(1 for pkt in packets_list if pkt.payload_type in VIDEO_TYPES)
    return video_cnt / len(packets_list)

# 14. Audio packets probability
# Audio packets probability: proportion of audio packets in the packets received in a MI, unit: packet/packet.
def audio_prob(packets_list):
    if not packets_list:
        return 0
    audio_cnt = sum(1 for pkt in packets_list if pkt.payload_type in AUDIO_TYPES)
    return audio_cnt / len(packets_list)

# 15. Probing packets probability
# Probing packets probability: proportion of probing packets in the packets received in a MI, unit: packet/packet.
def probe_prob(packets_list):
    if not packets_list:
        return 0
    probe_cnt = sum(1 for pkt in packets_list if pkt.payload_type in PROBE_TYPES)
    return probe_cnt / len(packets_list)

# 16. Received video bytes
def received_video_bytes(packets_list):
    return sum(pkt.payload_size for pkt in packets_list if pkt.payload_type in VIDEO_TYPES)

# 17. Received audio bytes
def received_audio_bytes(packets_list):
    return sum(pkt.payload_size for pkt in packets_list if pkt.payload_type in AUDIO_TYPES)

# 18. Payload type
def payload_type(packets_list):
    if not packets_list:
        return None
    return [pkt.payload_type for pkt in packets_list if pkt.payload_type is not None]

def receive_time(packets_list):
    if not packets_list:
        return None
    return [pk.receive_timestamp for pk in packets_list]

def send_time(packets_list):
    if not packets_list:
        return None
    return [pk.send_timestamp for pk in packets_list]

def packet_number(packets_list):
    if not packets_list:
        return None
    return [pk.sequence_number for pk in packets_list]