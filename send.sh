#!/bin/bash
modelName=$1

MODELDIR="./model"

runTestsOnModel() {
    modelName=$1
    # set active model
    MODEL="${MODELDIR}/${modelName}.pth"
    echo $MODEL >active_model
    docker run -d --rm --network host -v `pwd`:/app -w /app --name alphartc_sender --cap-add=NET_ADMIN challenge-env peerconnection_serverless sender_pyinfer.json
}

echo "Running tests for model: ${modelName}"
# 复制模型文件
cp BandwidthEstimator_${modelName}.py BandwidthEstimator.py
# 运行测试
runTestsOnModel "${modelName}"

