#!/bin/bash
date=$(date '+%d_%m_%Y_%H%M')
testid=0
MODELDIR="./model"
RESULTDIR="./results/${testid}/${date}"
DATA_LOGFILE="data.log"
rm -rf ${DATA_LOGFILE}

check_connection() {
  ss -tn sport = :8000 | grep -q ESTAB
}

wait_for_port_listen() {
  for i in {1..30}; do
    if ss -tnl | grep -q ':8000 '; then
      return 0
    fi
    sleep 1
  done
  return 1
}

runTestsOnModel() {
  modelName=$1
  resultsDir=$2
  rm -rf webrtc.log
  
  # set active model
  MODEL="${MODELDIR}/${modelName}.pth"
  echo $MODEL >active_model
  
  # 启动接收端
  docker run -d --rm --network host -v `pwd`:/app -w /app --name alphartc_receiver --cap-add=NET_ADMIN challenge-env peerconnection_serverless receiver_pyinfer.json
  sleep 1
  # 等待端口监听
  if ! wait_for_port_listen; then
    echo "接收端端口8000未监听，跳过本轮"
    docker stop alphartc_receiver >/dev/null 2>&1
    return
  fi
  # 远程登录并启动发送端
  ssh -p 2223 knw@202.120.36.216 "cd BoB_MIN && bash send.sh ${modelName}"
  # ssh -p 2223 knw@202.120.36.216 "cd BoB_MIN && docker run -d --rm --network host -v \$(pwd):/app -w /app --name alphartc_sender --cap-add=NET_ADMIN challenge-env peerconnection_serverless sender_pyinfer.json"
  # 等待连接建立，最多等待30秒
  for i in {1..30}; do
    if check_connection; then
      echo "连接已建立"
      start_time=$(date +%s)
      break
    fi
    sleep 1
  done

  if [ -z "$start_time" ]; then
    echo "连接未建立，跳过本轮测试"
    # 停止本地容器
    docker stop alphartc_receiver >/dev/null 2>&1
    # 停止远程容器（假设远程主机、用户名、端口已知）
    ssh -p 2223 knw@202.120.36.216 "docker stop alphartc_sender >/dev/null 2>&1"
    return
  fi

  # 等待连接结束
  while true; do
    if ! check_connection; then
      end_time=$(date +%s)
      echo "连接已结束"
      # 停止本地容器
      docker stop alphartc_receiver >/dev/null 2>&1
      # 停止远程容器（假设远程主机、用户名、端口已知）
      ssh -p 2223 knw@202.120.36.216 "docker stop alphartc_sender >/dev/null 2>&1"
      # 计算连接持续时间
      duration=$((end_time - start_time))
      echo "连接持续了 ${duration} 秒"
      break
    fi
    sleep 1
  done

  mv outvideo.yuv ${resultsDir}/outputvideo_${modelName}.yuv
  mv outaudio.wav ${resultsDir}/outaudio_${modelName}.wav
}

test_model_list=(
  # bob
  # gemini
  # hrcc
  # bob_heuristic
  bob1
)

for i in {1..5}
do
  for model in "${test_model_list[@]}"
  do
    modelResultDir=${RESULTDIR}_${model}_${i}
    echo "Running tests for model: $model, iteration: $i"
    # 清理旧的结果目录
    rm -rf $modelResultDir
    mkdir -p $modelResultDir
    # 复制对应的 BandwidthEstimator.py 文件
    cp BandwidthEstimator_${model}.py BandwidthEstimator.py
    # 运行测试
    runTestsOnModel "${model}" ${modelResultDir}
    # 移动sar日志文件
    if [ -f "${DATA_LOGFILE}" ]; then
      mv -f ${DATA_LOGFILE} ${modelResultDir}/
    else
      echo "警告：接收端未生成 data.log"
    fi
  done
done

