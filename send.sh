#!/bin/bash
# docker run -d --rm --network host -v `pwd`:/app -w /app --name alphartc_sender --cap-add=NET_ADMIN challenge-env peerconnection_serverless sender_pyinfer.json
modelName=$1
resultsDir=$2

date=$(date '+%d_%m_%Y_%H%M')
MODELDIR="./model"
RESULTDIR="results_${date}"
DATA_LOGFILE="data.log"
rm -rf ${DATA_LOGFILE}

runTestsOnModel() {
    modelName=$1
    resultsDir=$2
    rm -rf webrtc.log
    # set active model
    MODEL="${MODELDIR}/${modelName}".pth
    echo $MODEL >active_model
    docker run -d --rm --network host -v `pwd`:/app -w /app --name alphartc_sender --cap-add=NET_ADMIN challenge-env peerconnection_serverless sender_pyinfer.json
}

echo "Running tests for model: ${modelName}, results directory: ${resultsDir}"
# 清理旧的结果目录
rm -rf $resultsDir
mkdir -p $resultsDir
# 复制模型文件
cp BandwidthEstimator_${modelName}.py BandwidthEstimator.py
# 运行测试
runTestsOnModel "${modelName}" ${resultsDir}


