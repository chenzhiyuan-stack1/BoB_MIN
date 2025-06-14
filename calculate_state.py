import math

# 假设payload_type定义如下（请根据实际协议调整）
VIDEO_TYPES = {126}
AUDIO_TYPES = {97}
PROBE_TYPES = {100}
BASE_DELAY = 200  # ms

# 1. Receiving rate (bps)
def receiving_rate(packets_list):
    if not packets_list:
        return 0
    total_bytes = sum(pkt.size for pkt in packets_list)
    duration = packets_list[-1].receive_timestamp - packets_list[0].receive_timestamp
    duration = max(duration, 1)
    return total_bytes * 8 * 1000 / duration

# 2. Number of received packets
def num_received_packets(packets_list):
    return len(packets_list)

# 3. Received bytes
def received_bytes(packets_list):
    return sum(pkt.size for pkt in packets_list)

# 4. Queuing delay
def queuing_delay(packets_list, min_seen_delay):
    if not packets_list:
        return 0
    delays = [pkt.receive_timestamp - pkt.send_timestamp for pkt in packets_list]
    avg_delay = sum(delays) / len(delays)
    return avg_delay - min_seen_delay if min_seen_delay is not None else 0

# 5. Delay (avg delay - base delay)
def delay_minus_base(packets_list, base_delay=BASE_DELAY):
    if not packets_list:
        return 0
    delays = [pkt.receive_timestamp - pkt.send_timestamp for pkt in packets_list]
    avg_delay = sum(delays) / len(delays)
    return avg_delay - base_delay

# 6. Minimum seen delay (全局最小)
def min_seen_delay(packets_list, prev_min=None):
    if not packets_list:
        return prev_min if prev_min is not None else 0
    delays = [pkt.receive_timestamp - pkt.send_timestamp for pkt in packets_list]
    min_delay = min(delays)
    if prev_min is not None:
        return min(min_delay, prev_min)
    return min_delay

# 7. Delay ratio (avg delay / min delay in MI)
def delay_ratio(packets_list):
    if not packets_list:
        return 0
    delays = [pkt.receive_timestamp - pkt.send_timestamp for pkt in packets_list]
    avg_delay = sum(delays) / len(delays)
    min_delay = min(delays)
    return avg_delay / min_delay if min_delay > 0 else float('inf')

# 8. Delay average minimum difference
def delay_avg_min_diff(packets_list):
    if not packets_list:
        return 0
    delays = [pkt.receive_timestamp - pkt.send_timestamp for pkt in packets_list]
    avg_delay = sum(delays) / len(delays)
    min_delay = min(delays)
    return avg_delay - min_delay

# 9. Packet interarrival time (mean)
def mean_interarrival(packets_list):
    if len(packets_list) < 2:
        return 0
    arrival_times = [pkt.receive_timestamp for pkt in packets_list]
    interarrivals = [arrival_times[i] - arrival_times[i-1] for i in range(1, len(arrival_times))]
    return sum(interarrivals) / len(interarrivals)

# 10. Packet jitter (stddev of interarrival)
def packet_jitter(packets_list):
    if len(packets_list) < 2:
        return 0
    arrival_times = [pkt.receive_timestamp for pkt in packets_list]
    interarrivals = [arrival_times[i] - arrival_times[i-1] for i in range(1, len(arrival_times))]
    mean_ia = sum(interarrivals) / len(interarrivals)
    if len(interarrivals) < 2:
        return 0
    return math.sqrt(sum((x - mean_ia) ** 2 for x in interarrivals) / (len(interarrivals)-1))

# 11. Packet loss ratio
def packet_loss_ratio(packets_list):
    seqs = [pkt.sequence_number for pkt in packets_list]
    if not seqs:
        return 0
    expected = max(seqs) - min(seqs) + 1
    received = len(seqs)
    return 1 - received / expected if expected > 0 else 0

# 12. Average number of lost packets (每次丢包的平均丢包数)
def avg_lost_pkts(packets_list):
    seqs = sorted(pkt.sequence_number for pkt in packets_list)
    lost_counts = []
    for i in range(1, len(seqs)):
        gap = seqs[i] - seqs[i-1] - 1
        if gap > 0:
            lost_counts.append(gap)
    return sum(lost_counts) / len(lost_counts) if lost_counts else 0

# 13. Video packets probability
def video_prob(packets_list):
    if not packets_list:
        return 0
    video_cnt = sum(1 for pkt in packets_list if pkt.payload_type in VIDEO_TYPES)
    return video_cnt / len(packets_list)

# 14. Audio packets probability
def audio_prob(packets_list):
    if not packets_list:
        return 0
    audio_cnt = sum(1 for pkt in packets_list if pkt.payload_type in AUDIO_TYPES)
    return audio_cnt / len(packets_list)

# 15. Probing packets probability
def probe_prob(packets_list):
    if not packets_list:
        return 0
    probe_cnt = sum(1 for pkt in packets_list if pkt.payload_type in PROBE_TYPES)
    return probe_cnt / len(packets_list)