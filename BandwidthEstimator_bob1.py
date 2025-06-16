#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
from utils.packet_info import PacketInfo
from utils.packet_record import PacketRecord
#from deep_rl.actor_critic import ActorCritic
from deep_rl.ppo_AC import Actor
from deep_rl.ppo_AC import Critic
from collections import deque
from BandwidthEstimator_heuristic import HeuristicEstimator
import logging

import json
import calculate_state
import record_slice
import math

UNIT_M = 1000000
MAX_BANDWIDTH_MBPS = 8
MIN_BANDWIDTH_MBPS = 0.01
LOG_MAX_BANDWIDTH_MBPS = np.log(MAX_BANDWIDTH_MBPS)
LOG_MIN_BANDWIDTH_MBPS = np.log(MIN_BANDWIDTH_MBPS)
#global FactorH
FactorH = 1.10

logging.basicConfig(filename='bandwidth_estimator.log', level=logging.DEBUG)

def liner_to_log(value):
    # from 10kbps~8Mbps to 0~1
    value = np.clip(value / UNIT_M, MIN_BANDWIDTH_MBPS, MAX_BANDWIDTH_MBPS)
    log_value = np.log(value)
    return (log_value - LOG_MIN_BANDWIDTH_MBPS) / (LOG_MAX_BANDWIDTH_MBPS - LOG_MIN_BANDWIDTH_MBPS)


def log_to_linear(value):
    # from 0~1 to 10kbps to 8Mbps
    value = np.clip(value, 0, 1)
    log_bwe = value * (LOG_MAX_BANDWIDTH_MBPS - LOG_MIN_BANDWIDTH_MBPS) + LOG_MIN_BANDWIDTH_MBPS
    return np.exp(log_bwe) * UNIT_M

def load_active_model(active_model_file='active_model'):
    with open(active_model_file, 'r') as f:
        try:
            model=f.read().strip()
            logging.debug("Using model="+model)
        except Exception as ex:
            logging.debug("Couldn't find active model using default value! Exception:" + ex)
            model='./model/bob.pth'
    return model

class Estimator(object):
    def __init__(self, model_path=load_active_model(), step_time=60): # Make sure to push model and change the name of the model
        # model parameters
        state_dim = 11
        action_dim = 4
        # the std var of action distribution
        exploration_param = 0.05
        actionSelected = []
        sumall = 0
        FactorH = 1.1
        # load model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy = Actor(state_dim, action_dim, exploration_param, self.device).to(self.device)
        self.value =  Critic(state_dim, action_dim, exploration_param, self.device).to(self.device)
        self.policy.load_state_dict(torch.load(model_path))#(self.policy.state_dict())
        #self.value.load_state_dict(torch.load(model_path))
        self.value.load_state_dict(self.value.state_dict())
        #self.model = ActorCritic(state_dim, action_dim, exploration_param, self.device).to(self.device)
        #self.model.load_state_dict(torch.load(model_path))
        # the model to get the input of model
        self.packet_record = PacketRecord()
        self.packet_record.reset()
        self.step_time = step_time
        self.first_arrival_time = 0
        self.last_arrival_time = 0
        # init
        #states = [0.0, 0.0, 0.0, 0.
        self.bandwdith_list_state =  deque([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        states = np.append([0.0, 0.0, 0.0],self.bandwdith_list_state)
        '''
        torch_tensor_states = torch.FloatTensor(torch.Tensor(states).reshape(1, -1)).to(self.device)
        action, action_logprobs, value = self.model.forward(torch_tensor_states)
        self.bandwidth_prediction = log_to_linear(action)
        self.last_call = "init"
        '''
        torch_tensor_states = torch.FloatTensor(torch.Tensor(states).reshape(1, -1)).to(self.device)
        action, action_logprobs = self.policy.forward(torch_tensor_states)
        value = self.value.forward(torch_tensor_states)

        softmax_action = torch.exp(action)
        action = softmax_action.detach().reshape(1, -1)
        sumall = np.sum(action[0].tolist())
        actionSelected = action[0].tolist()/sumall
        actionSelected = np.random.choice(4,p=actionSelected)

        self.bandwidth_prediction = log_to_linear(action[0][actionSelected])
        self.last_call = "init"

        #heuristic
        self.heuristic_estimator = HeuristicEstimator()
        self.delay = 0
        self.loss_ratio = 0
        
        self.mi_idx = 0  # 当前MI编号
        self.global_min_delay = None  # 全局最小延迟
        self.packets_list = []

    
    def report_states(self, stats: dict):
        '''
        stats is a dict with the following items
        {
            "send_time_ms": uint,
            "arrival_time_ms": uint,
            "payload_type": int,
            "sequence_number": uint,
            "ssrc": int,
            "padding_length": uint,
            "header_length": uint,
            "payload_size": uint
        }
        '''

        if self.last_arrival_time != 0:
            self.step_time = stats["arrival_time_ms"] - self.last_arrival_time
        else:
            self.first_arrival_time = stats["arrival_time_ms"]
        self.last_arrival_time = stats["arrival_time_ms"]

        self.last_call = "report_states"
        # clear data
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
        packet_info.bandwidth_prediction = self.bandwidth_prediction

        self.packet_record.on_receive(packet_info)
        self.heuristic_estimator.report_states(stats)
        
        self.packets_list.append(packet_info)

    def record_mi_state(
        self,
        packets_list,
        bandwidth_estimation,
        mi_idx,
        min_seen_delay_global,
        audio_path,
        video_path,
        video_width,
        video_height,
        data_log_path="data.jsonl"  # 每个MI的记录存储路径，默认是data.jsonl
    ):
        """
        记录每个MI的网络状态、动作（带宽）、音视频信息到data.jsonl，格式化为一行一个json，便于后续处理。
        """
        # 1. 计算网络状态特征
        state = {
            "receiving_rate": calculate_state.receiving_rate(packets_list),
            "num_received_packets": calculate_state.num_received_packets(packets_list),
            "received_bytes": calculate_state.received_bytes(packets_list),
            "queuing_delay": calculate_state.queuing_delay(packets_list, min_seen_delay_global),
            "delay_minus_base": calculate_state.delay_minus_base(packets_list),
            "min_seen_delay": calculate_state.min_seen_delay(packets_list, min_seen_delay_global),
            "delay_ratio": calculate_state.delay_ratio(packets_list),
            "delay_avg_min_diff": calculate_state.delay_avg_min_diff(packets_list),
            "mean_interarrival": calculate_state.mean_interarrival(packets_list),
            "packet_jitter": calculate_state.packet_jitter(packets_list),
            "packet_loss_ratio": calculate_state.packet_loss_ratio(packets_list),
            "avg_lost_pkts": calculate_state.avg_lost_pkts(packets_list),
            "video_prob": calculate_state.video_prob(packets_list),
            "audio_prob": calculate_state.audio_prob(packets_list),
            "probe_prob": calculate_state.probe_prob(packets_list),
            "received_video_bytes": calculate_state.received_video_bytes(packets_list),
            "received_audio_bytes": calculate_state.received_audio_bytes(packets_list),
            "payload_type": calculate_state.payload_type(packets_list),
        }

        # 2. 动作
        action = {
            "bandwidth_estimation": bandwidth_estimation
        }

        # # 3. 音视频信息
        # audio_info = {
        #     "audio_path": audio_path,
        #     "audio_duration": record_slice.get_wav_duration(audio_path)
        # }
        # video_info = {
        #     "video_path": video_path,
        #     "video_frame_count": record_slice.get_yuv_frame_count(video_path, video_width, video_height)
        # }

        # 4. 组合为一条记录
        mi_record = {
            "mi_idx": mi_idx,
            "state": state,
            "action": action,
            # "audio_info": audio_info,
            # "video_info": video_info
        }

        # 5. 写入data.jsonl（每行一个json，便于后续处理）
        with open(data_log_path, "a") as f:
            f.write(json.dumps(mi_record, ensure_ascii=False) + "\n")

    def get_estimated_bandwidth(self)->int:
        if self.last_call and self.last_call == "report_states":
            self.last_call = "get_estimated_bandwidth"
            # calculate state
            global FactorH
            states = []
            receiving_rate = self.packet_record.calculate_receiving_rate(interval=self.step_time)
            states.append(liner_to_log(receiving_rate))
            previousDelay = self.delay
            self.delay = self.packet_record.calculate_average_delay(interval=self.step_time)
            #states.append(min(self.delay/1000, 1))
            states.append(self.delay/1000)
            previousLossRatio = self.loss_ratio
            self.loss_ratio = self.packet_record.calculate_loss_ratio(interval=self.step_time)
            states.append(self.loss_ratio)

            heuristic_prediction, heuristic_overuse_flag = self.heuristic_estimator.get_estimated_bandwidth()
            heuristic_prediction = heuristic_prediction * FactorH #1.10 #FactorH #1.10

            for l in self.bandwdith_list_state:
                states.append(l)
                
            #latest_prediction = self.packet_record.calculate_latest_prediction()
            BW_state = liner_to_log(heuristic_prediction)
            self.bandwdith_list_state.popleft()
            self.bandwdith_list_state.append(BW_state)

            #states.append(liner_to_log(heuristic_prediction))
            #delay_interval = self.packet_record.calculate_delay_interval(interval=self.step_time)
            #states.append(min(delay_interval/1000,1))
            # make the states for model
            torch_tensor_states = torch.FloatTensor(torch.Tensor(states).reshape(1, -1)).to(self.device)
            # get model output
            #action, action_logprobs, value = self.model.forward(torch_tensor_states)
            action, action_logprobs = self.policy.forward(torch_tensor_states)
            value = self.value.forward(torch_tensor_states)
            softmax_action = torch.exp(action)
            action = softmax_action.detach().reshape(1, -1)
            sumall = np.sum(action[0].tolist())
            actionSelected = action[0].tolist()/sumall
            MinactionSelected = actionSelected
            MinactionSelected = np.where(MinactionSelected == np.amin(MinactionSelected))
            actionSelected = np.random.choice(4,p=actionSelected)
            # update prediction of bandwidth by using action
            learningBasedBWE = log_to_linear(pow(2, (2 * action[0][actionSelected] - 1))).item()
            #MinactionSelected = np.where(MinactionSelected == np.amin(action[0].tolist()/sumall)) 
            FactorH = 1 - (action[0][MinactionSelected]).item()/2  # 1.02 - (action[0][actionSelected]).item()/2  # test with directly 0.8 and with min action +-  #pow(2, (2 * action[0][actionSelected] - 1)).item() + 1
            #FactorH = (action[0][actionSelected]).item() + 0.8
            self.bandwidth_prediction = learningBasedBWE
            isHeuristicUsed=False

            diff_predictions = abs(int(self.bandwidth_prediction) - int(heuristic_prediction))
            average_predictions = (int(self.bandwidth_prediction) + int(heuristic_prediction)) / 2
            percentage = diff_predictions / average_predictions
            if percentage >= 0.3:
                self.bandwidth_prediction = heuristic_prediction
                if self.delay - previousDelay < 200:   
                    FactorH = (action[0][actionSelected]).item() + 0.85
                isHeuristicUsed=True

            #self.bandwidth_prediction = log_to_linear(percentage)
            #self.bandwidth_prediction = min(self.bandwidth_prediction,MAX_BANDWIDTH_MBPS*UNIT_M)
            #self.bandwidth_prediction = max(self.bandwidth_prediction,MIN_BANDWIDTH_MBPS*UNIT_M)

            self.heuristic_estimator.change_bandwidth_estimation(self.bandwidth_prediction)
            logging.debug("time:"+str(self.last_arrival_time - self.first_arrival_time)+" actual_bw:"+str(receiving_rate)+" predicted_bw:"+str(self.bandwidth_prediction)+ " isHeuristicUsed:"+str(isHeuristicUsed)+ " heuristic_overuse_flag:"+str(heuristic_overuse_flag)+ " HeuristicBW:" +str(heuristic_prediction) + " learningBW:" + str(learningBasedBWE)+" Actions:"+str(action)+" SelectedActionIdx:"+str(actionSelected)+" SeletedAction:"+str(action[0][actionSelected])+" Percentage:"+str(percentage)+" FactorH:"+str(FactorH))
        
        # 维护global_min_delay
        if self.global_min_delay is None:
            self.global_min_delay = calculate_state.min_seen_delay(self.packets_list, 0)
        else:
            current_min = calculate_state.min_seen_delay(self.packets_list, self.global_min_delay)
            if current_min < self.global_min_delay:
                self.global_min_delay = current_min
        
        # 记录data
        self.record_mi_state(
            packets_list=self.packets_list,
            bandwidth_estimation=self.bandwidth_prediction,
            mi_idx=self.mi_idx,
            min_seen_delay_global=self.global_min_delay,
            audio_path="outaudio.wav",
            video_path="outvideo.yuv",
            video_width=320,
            video_height=240,
            data_log_path="data.jsonl"
        )
        
        self.mi_idx += 1  # MI编号自增
        
        self.packets_list = []  # 清空当前MI的包列表
        
        return self.bandwidth_prediction