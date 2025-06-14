docker run -d --rm -v `pwd`:/app -w /app --name alphartc_pyinfer opennetlab.azurecr.io/challenge-env peerconnection_serverless receiver_pyinfer.json
sleep 1
docker exec alphartc_pyinfer peerconnection_serverless sender_pyinfer.json

# docker run -d --rm --network host -v `pwd`:/app -w /app --name alphartc_receiver --cap-add=NET_ADMIN challenge-env peerconnection_serverless receiver_pyinfer.json
# sleep 5
# 远程登录并启动发送端
# ssh -p 2223 knw@202.120.36.216 "cd /home/knw/BoB && docker run -d --rm --network host -v \$(pwd):/app -w /app --name alphartc_sender --cap-add=NET_ADMIN challenge-env peerconnection_serverless sender_pyinfer.json"

# check_connection() {
#   ss -tn sport = :8000 | grep -q ESTAB
# }

# for i in {1..30}; do
#     if check_connection; then
#         echo "连接已建立"
#         start_time=$(date +%s)
#         break
#     fi
#     sleep 1
# done

# if [ -z "$start_time" ]; then
#     echo "连接未建立，跳过本轮测试"
#     # 停止本地容器
#     docker stop alphartc_receiver >/dev/null 2>&1
#     # 停止远程容器（假设远程主机、用户名、端口已知）
#     ssh -p 2223 knw@202.120.36.216 "docker stop alphartc_pyinfer >/dev/null 2>&1"
#     exit 1
# fi

# # 等待连接结束
# while true; do
#     if ! check_connection; then
#         end_time=$(date +%s)
#         echo "连接已结束"
#         duration=$((end_time - start_time))
#         echo "连接持续了 ${duration} 秒"
#         break
#     fi
#     sleep 1
# done