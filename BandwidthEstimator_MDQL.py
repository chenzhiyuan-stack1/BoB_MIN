# -*- coding: utf-8 -*-
import torch
import numpy as np
from collections import deque
import logging
import json
import math
import os

import calculate_state
from utils.packet_info import PacketInfo
from utils.packet_record import PacketRecord
from BandwidthEstimator_heuristic2 import HeuristicEstimator
from diffusion.ql_diffusion_e2e import Diffusion_QL as Agent
from diffusion.norm_vector import NORMAL_VECTOR
from dataset.result2dataset import update_and_get_observation


UNIT_M = 1e6
FactorH = 1.10

logging.basicConfig(filename='bandwidth_estimator.log', level=logging.DEBUG)

def load_active_model(active_model_file='active_model'):
    with open(active_model_file, 'r') as f:
        try:
            model=f.read().strip()
            logging.debug("Using model="+model)
        except Exception as ex:
            logging.debug("Couldn't find active model using default value! Exception:" + ex)
            model='./model/new.pth'
    return model

class Estimator(object):
    def __init__(self, model_path=None, step_time=200):
        if model_path is None:
            model_path = load_active_model()
        
        self.state_dim_t = 11
        self.state_window_size = 6
        self.obs_dim = self.state_dim_t * self.state_window_size
        self.action_dim = 1
        self.max_action_mbps = 20.0

        self.agent = Agent(
            state_dim=self.obs_dim,
            action_dim=self.action_dim,
            max_action=self.max_action_mbps,
        )
        try:
            # 加载到正确的设备
            self.agent.actor.load_state_dict(torch.load(model_path))
            self.agent.actor.eval()
            logging.info(f"Successfully loaded Diffusion-QL model from: {model_path}")
        except Exception as e:
            logging.error(f"Failed to load Diffusion-QL model from {model_path}. Error: {e}")
            # 程序可以继续，但RL部分将输出0
            
        self.state_history = deque([np.zeros(self.state_dim_t)] * self.state_window_size, maxlen=self.state_window_size)
        
        # 保留启发式方法作为对比和混合策略
        self.heuristic_estimator = HeuristicEstimator()
        
        # 其他状态变量
        self.packet_record = PacketRecord()
        self.packet_record.reset()
        self.step_time = step_time
        self.last_arrival_time = 0
        self.bandwidth_prediction = 2 * UNIT_M
        self.last_call = "init"
        
        self.mi_idx = 0
        self.global_min_delay = float("inf")
        self.packets_list = []
        
        self.delay = 0  # 新增：用于动态FactorH调整
        self.previousDelay = 0  # 新增：用于动态FactorH调整

    def report_states(self, stats: dict):
        if self.last_arrival_time != 0:
            self.step_time = stats["arrival_time_ms"] - self.last_arrival_time
        self.last_arrival_time = stats["arrival_time_ms"]
        self.last_call = "report_states"
        
        packet_info = PacketInfo()
        packet_info.payload_type = stats["payload_type"]
        packet_info.ssrc = stats["ssrc"]
        packet_info.sequence_number = stats["sequence_number"]
        packet_info.send_timestamp = stats["send_time_ms"]
        packet_info.receive_timestamp = stats["arrival_time_ms"]
        packet_info.padding_length = stats["padding_length"]
        packet_info.header_length = stats["header_length"]
        packet_info.payload_size = stats["payload_size"]
        packet_info.size = packet_info.header_length + packet_info.payload_size + packet_info.padding_length
        packet_info.bandwidth_prediction = int(self.bandwidth_prediction)

        self.packet_record.on_receive(packet_info)
        self.heuristic_estimator.report_states(stats)
        self.packets_list.append(packet_info)

    def get_estimated_bandwidth(self) -> int:
        if not (self.last_call and self.last_call == "report_states"):
            return int(self.bandwidth_prediction)

        self.last_call = "get_estimated_bandwidth"
        
        # --- 1. 统一计算当前决策周期的所有状态指标 ---
        # 这个字典将用于模型输入和日志记录，避免重复计算
        current_state_dict = {
            "receiving_rate": calculate_state.receiving_rate(self.packets_list),
            "num_received_packets": calculate_state.num_received_packets(self.packets_list),
            "received_bytes": calculate_state.received_bytes(self.packets_list),
            "queuing_delay": calculate_state.queuing_delay(self.packets_list, self.global_min_delay),
            "delay_minus_base": calculate_state.delay_minus_base(self.packets_list),
            "min_seen_delay": calculate_state.min_seen_delay(self.packets_list, self.global_min_delay),
            "delay_ratio": calculate_state.delay_ratio(self.packets_list),
            "delay_avg_min_diff": calculate_state.delay_avg_min_diff(self.packets_list),
            "mean_interarrival": calculate_state.mean_interarrival(self.packets_list),
            "packet_jitter": calculate_state.packet_jitter(self.packets_list),
            "packet_loss_ratio": calculate_state.packet_loss_ratio(self.packets_list),
        }
        
        # --- 维护 delay 和 previousDelay ---
        self.previousDelay = self.delay
        self.delay = current_state_dict.get("delay_avg_min_diff", 0.0)
        
        # --- 2. 构建模型输入 (obs) 并进行预测 ---
        state_keys_for_model = [
            "receiving_rate", "num_received_packets", "received_bytes", "queuing_delay",
            "delay_minus_base", "min_seen_delay", "delay_ratio", "delay_avg_min_diff",
            "mean_interarrival", "packet_jitter", "packet_loss_ratio"
        ]
        current_state_t = np.array([current_state_dict.get(k, 0.0) for k in state_keys_for_model])
        
        # 构建 66 维的 observation
        obs = update_and_get_observation(self.state_history, current_state_t)
        # 归一化并转换为 Tensor
        obs_normalized = obs * NORMAL_VECTOR
        obs_tensor = torch.tensor(obs_normalized.reshape(1, -1), dtype=torch.float32)
        # 使用 Diffusion-QL 模型预测带宽
        with torch.no_grad():
            action_bps = self.agent.actor.sample(obs_tensor).cpu().numpy().flatten()[0]
        learningBasedBWE = action_bps
        
        # --- 3. 结合启发式方法，得到最终预测值 (沿用旧逻辑) ---
        global FactorH
        heuristic_prediction, heuristic_overuse_flag = self.heuristic_estimator.get_estimated_bandwidth()
        # heuristic_prediction = heuristic_prediction * FactorH

        self.bandwidth_prediction = learningBasedBWE
        isHeuristicUsed = False
        
        # 动态调整FactorH（仿照bob1）
        try:
            FactorH = 1 - (action_bps / (self.max_action_mbps * UNIT_M)) / 2
        except Exception as e:
            logging.warning(f"FactorH dynamic adjustment failed: {e}")
        
        # 估计冷启动的时候，模型输出是NAN
        if math.isnan(learningBasedBWE) or math.isnan(heuristic_prediction):
            logging.error(f"NaN detected! learningBasedBWE={learningBasedBWE}, heuristic_prediction={heuristic_prediction}")
            # 兜底：用启发式或默认值
            self.bandwidth_prediction = heuristic_prediction if not math.isnan(heuristic_prediction) else 2 * UNIT_M
            heuristic_prediction = self.bandwidth_prediction
        
        diff_predictions = abs(int(self.bandwidth_prediction) - int(heuristic_prediction))
        average_predictions = (int(self.bandwidth_prediction) + int(heuristic_prediction)) / 2
        percentage = diff_predictions / average_predictions
        if percentage >= 0.3: # 如果差异过大，信任启发式方法
            self.bandwidth_prediction = heuristic_prediction
            if self.delay - self.previousDelay < 200:
                FactorH = (action_bps / (self.max_action_mbps * UNIT_M)) + 0.85
            isHeuristicUsed = True

        # 确保带宽在合理范围内
        self.bandwidth_prediction = np.clip(self.bandwidth_prediction, 0.1 * UNIT_M, self.max_action_mbps * UNIT_M)
        self.heuristic_estimator.change_bandwidth_estimation(self.bandwidth_prediction)
        
        logging.debug(f"time:{(self.last_arrival_time or 0)} "
                      f"actual_bw:{current_state_dict['receiving_rate']:.0f} "
                      f"predicted_bw:{self.bandwidth_prediction:.0f} "
                      f"isHeuristicUsed:{isHeuristicUsed} "
                      f"heuristic_overuse_flag:{heuristic_overuse_flag} "
                      f"HeuristicBW:{heuristic_prediction:.0f} "
                      f"learningBW:{learningBasedBWE:.0f} "
                      f"Percentage:{percentage:.2f} FactorH:{FactorH:.2f}")

        # --- 4. 记录当前决策周期的完整信息 (日志) ---
        # 维护全局最小延迟
        self.global_min_delay = min(self.global_min_delay, current_state_dict['min_seen_delay'])

        mi_record = {
            "mi_idx": self.mi_idx,
            "state": {**current_state_dict, **self._get_full_packet_info(), "isHeuristicUsed": isHeuristicUsed},
            "action": {"bandwidth_estimation": int(self.bandwidth_prediction)}
        }

        try:
            with open("data.jsonl", "a") as f:
                f.write(json.dumps(mi_record, ensure_ascii=False) + "\n")
        except Exception as e:
            logging.error(f"Failed to write to data.jsonl: {e}")

        self.mi_idx += 1
        self.packets_list = [] # 清空当前MI的包列表
        
        return int(self.bandwidth_prediction)

    def _get_full_packet_info(self) -> dict:
        return {
            "avg_lost_pkts": calculate_state.avg_lost_pkts(self.packets_list),
            "video_prob": calculate_state.video_prob(self.packets_list),
            "audio_prob": calculate_state.audio_prob(self.packets_list),
            "probe_prob": calculate_state.probe_prob(self.packets_list),
            "received_video_bytes": calculate_state.received_video_bytes(self.packets_list),
            "received_audio_bytes": calculate_state.received_audio_bytes(self.packets_list),
            "payload_type": calculate_state.payload_type(self.packets_list),
            "send_time": calculate_state.send_time(self.packets_list),
            "receive_time": calculate_state.receive_time(self.packets_list),
            "sequence_number": calculate_state.packet_number(self.packets_list),
            "all_payload_type": calculate_state.all_payload_type(self.packets_list),
            "all_sequence_number": calculate_state.all_sequence_number(self.packets_list),
            "all_send_timestamp": calculate_state.all_send_timestamp(self.packets_list),
            "all_ssrc": calculate_state.all_ssrc(self.packets_list),
            "all_padding_length": calculate_state.all_padding_length(self.packets_list),
            "all_header_length": calculate_state.all_header_length(self.packets_list),
            "all_receive_timestamp": calculate_state.all_receive_timestamp(self.packets_list),
            "all_payload_size": calculate_state.all_payload_size(self.packets_list),
            "all_bandwidth_prediction": calculate_state.all_bandwidth_prediction(self.packets_list),
        }