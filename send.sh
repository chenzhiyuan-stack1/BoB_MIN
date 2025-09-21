#!/bin/bash
modelName=$1
video_path=$2
height=$3
width=$4
fps=$5
audio_path=$6

MODELDIR="./model"

runTestsOnModel() {
    modelName=$1
    # set active model
    MODEL="${MODELDIR}/${modelName}.pth"
    echo $MODEL >active_model

    # 清理上一轮发送端的文件和日志
    rm -rf webrtc.log

    # 修改 sender_pyinfer.json 的 video_file 和 audio_file 配置
    jq --arg vpath "$video_path" --argjson height $height --argjson width $width --argjson fps $fps \
       --arg apath "$audio_path" \
      '.video_source.video_file.file_path=$vpath
       | .video_source.video_file.height=$height
       | .video_source.video_file.width=$width
       | .video_source.video_file.fps=$fps
       | .video_source.video_file.enabled=true
       | .audio_source.audio_file.file_path=$apath
       | .audio_source.audio_file.enabled=true' \
      sender_pyinfer.json > sender_pyinfer_tmp.json && mv sender_pyinfer_tmp.json sender_pyinfer_online.json

    docker run -d --network host -v `pwd`:/app -w /app --name alphartc_sender --cap-add=NET_ADMIN challenge-env peerconnection_serverless sender_pyinfer_online.json
}

echo "Running tests for model: ${modelName}"
echo "Video path: ${video_path}, Height: ${height}, Width: ${width}, FPS: ${fps}"
echo "Audio path: ${audio_path}"
cp BandwidthEstimator_${modelName}.py BandwidthEstimator.py
runTestsOnModel "${modelName}"