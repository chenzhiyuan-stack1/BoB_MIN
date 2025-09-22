#!/bin/bash
date=$(date '+%d_%m_%Y_%H%M')
testid=$1
MODELDIR="./model"
RESULTDIR="./results/${testid}/${date}"
DATA_LOGFILE="data.jsonl"
rm -rf ${DATA_LOGFILE}

check_connection() {
  # 检查是否有到8000端口的已建立连接
  ss -tn sport = :8000 | grep -q ESTAB
}

wait_for_port_listen() {
  echo "等待接收端端口 8000 监听..."
  for j in {1..30}; do
    if ss -tnl | grep -q ':8000 '; then
      echo "端口 8000 已监听。"
      return 0
    fi
    sleep 1
  done
  echo "错误: 等待端口 8000 监听超时。"
  return 1
}

# 网络 profile 列表
tc_profiles=(
  test
)

# 随机选video和audio，并远程启动发送端
runTestsOnModel() {
  modelName=$1
  resultsDir=$2

  # 随机选一个视频
  video=${video_list[$((RANDOM % ${#video_list[@]}))]}
  params=(${video_params[$video]})
  video_path=${params[0]}
  height=${params[1]}
  width=${params[2]}
  fps=${params[3]}

  # 随机选一个音频
  audio=${audio_list[$((RANDOM % ${#audio_list[@]}))]}
  audio_path=${audio_params[$audio]}

  echo "Selected video: $video (path: $video_path, height: $height, width: $width, fps: $fps)"
  echo "Selected audio: $audio (path: $audio_path)"

  # 清理上一轮接收端的文件和日志
  rm -rf webrtc.log outvideo.yuv outaudio.wav

  # set active model
  MODEL="${MODELDIR}/${modelName}.pth"
  echo $MODEL >active_model

  # 修改 receiver_pyinfer.json 的配置
  echo "Configuring receiver for video width: $width, height: $height, fps: $fps"
  jq --argjson width $width --argjson height $height --argjson fps $fps \
    '.save_to_file.video.width=$width | .save_to_file.video.height=$height | .save_to_file.video.fps=$fps' \
    receiver_pyinfer.json > receiver_pyinfer_tmp.json && mv receiver_pyinfer_tmp.json receiver_pyinfer_online.json

  # 启动接收端容器
  echo "Starting receiver..."
  docker run -d --network host -v `pwd`:/app -w /app --name alphartc_receiver --cap-add=NET_ADMIN challenge-env-tc peerconnection_serverless receiver_pyinfer_online.json
  
  if ! wait_for_port_listen; then
    echo "接收端端口8000未监听，跳过本轮"
    docker stop alphartc_receiver >/dev/null 2>&1 && docker rm alphartc_receiver >/dev/null 2>&1
    return
  fi

  # 远程启动发送端
  ssh -p 2223 knw@202.120.36.216 "cd BoB_MIN && bash send.sh ${modelName} \"$video_path\" $height $width $fps \"$audio_path\""

  # 等待连接建立
  echo "等待连接建立..."
  for k in {1..30}; do
    if check_connection; then
      echo "连接已建立"
      start_time=$(date +%s)
      break
    fi
    sleep 1
  done

  if [ -z "$start_time" ]; then
    echo "连接未建立，跳过本轮测试"
    docker stop alphartc_receiver >/dev/null 2>&1 && docker rm alphartc_receiver >/dev/null 2>&1
    ssh -p 2223 knw@202.120.36.216 "docker stop alphartc_sender >/dev/null 2>&1 && docker rm alphartc_sender >/dev/null 2>&1"
    return
  fi

  # ==================== TC 核心修改 ====================
  # 随机选一个 tc profile
  tc_profile=${tc_profiles[$((RANDOM % ${#tc_profiles[@]}))]}
  echo "选用链路 profile: $tc_profile"
  
  # 在宿主机上用 sudo 启动 tc 策略脚本，并在后台运行
  echo "--- 在宿主机上启动 TC 策略 ---"
  sudo bash /home/min414/data2/BoB_MIN/tc_profiles/tc_policy_min1.sh "/home/min414/data2/BoB_MIN/tc_profiles/${tc_profile}" &
  # 保存后台脚本的进程ID (PID)，以便后续杀死它
  TC_POLICY_PID=$!
  echo "TC 策略脚本已启动，PID: $TC_POLICY_PID"
  # ======================================================

  # 等待连接结束
  while true; do
    if ! check_connection; then
      end_time=$(date +%s)
      echo "连接已结束"
      
      # ==================== TC 清理核心修改 ====================
      # 杀死后台运行的 tc 策略脚本
      echo "--- 停止 TC 策略脚本 (PID: $TC_POLICY_PID) ---"
      # 检查进程是否存在，然后杀死
      if ps -p $TC_POLICY_PID > /dev/null; then
         sudo kill -9 $TC_POLICY_PID
         echo "TC 策略进程已停止。"
      else
         echo "TC 策略进程已自行退出。"
      fi
      
      # 在宿主机上运行清理脚本，确保清理干净
      echo "--- 在宿主机上清理所有 TC 规则 ---"
      sudo bash /home/min414/data2/BoB_MIN/tc_profiles/tc_clear_min.sh
      # ==========================================================

      # 停止并删除容器
      if docker ps -a | grep -q alphartc_receiver; then
        echo "停止并删除 receive 容器..."
        docker stop alphartc_receiver >/dev/null 2>&1
        docker rm alphartc_receiver >/dev/null 2>&1
      fi
      # 远程停止发送端容器
      ssh -p 2223 knw@202.120.36.216 "docker stop alphartc_sender >/dev/null 2>&1 && docker rm alphartc_sender >/dev/null 2>&1"
      
      duration=$((end_time - start_time))
      echo "连接持续了 ${duration} 秒"
      break
    fi
    sleep 1
  done

  # 收集结果文件
  mv outvideo.yuv ${resultsDir}/outputvideo_${modelName}_${video}_${audio}_${tc_profile}.yuv
  mv outaudio.wav ${resultsDir}/outaudio_${modelName}_${video}_${audio}_${tc_profile}.wav
  mv webrtc.log ${resultsDir}/webrtc_receive_${modelName}_${video}_${audio}_${tc_profile}.log
  # 远程收集发送端日志
  ssh -p 2223 knw@202.120.36.216 "cd BoB_MIN && rm -rf ${resultsDir} && mkdir -p ${resultsDir} && mv webrtc.log ${resultsDir}/webrtc_send_${modelName}_${video}_${audio}_${tc_profile}.log"
}

# ==============================================================================
# 主循环和配置
# ==============================================================================

test_model_list=(
  heuristic2
)

declare -A video_params
video_list=(
  test
  akiyo_qcif
  bowing_cif
  bus_cif
  carphone_cif
  claire_qcif
  coastguard_qcif
  container_qcif
)
video_params[test]="testmedia/test.yuv 240 320 10"
video_params[akiyo_qcif]="testmedia/akiyo_qcif.yuv 144 176 30"
video_params[bowing_cif]="testmedia/bowing_cif.yuv 288 352 30"
video_params[bus_cif]="testmedia/bus_cif.yuv 288 352 30"
video_params[carphone_cif]="testmedia/carphone_cif.yuv 288 352 30"
video_params[claire_qcif]="testmedia/claire_qcif.yuv 144 176 30"
video_params[coastguard_qcif]="testmedia/coastguard_qcif.yuv 144 176 30"
video_params[container_qcif]="testmedia/container_qcif.yuv 144 176 30"

declare -A audio_params
audio_list=(
  test
  test_30
)
audio_params[test]="testmedia/test.wav"
audio_params[test_30]="testmedia/test_30.wav"

for i in {1..2}
do
  for model in "${test_model_list[@]}"
  do
    modelResultDir=${RESULTDIR}_${model}_${i}
    echo "Running tests for model: $model, iteration: $i"
    rm -rf $modelResultDir
    mkdir -p $modelResultDir
    cp BandwidthEstimator_${model}.py BandwidthEstimator.py
    runTestsOnModel "${model}" ${modelResultDir}
    if [ -f "${DATA_LOGFILE}" ]; then
      mv -f ${DATA_LOGFILE} ${modelResultDir}/
    else
      echo "警告：接收端未生成 data.jsonl"
    fi
  done
done