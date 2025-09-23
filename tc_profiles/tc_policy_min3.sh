#!/bin/bash
# TC="/sbin/tc"
TC="tc"
INTERFACE_1=enp0s31f6
PORT_1=8000
FILE_1=$1

if [ -z "$FILE_1" ]; then
  echo "policy file name has to be specified"
  exit 1;
fi

parsePolicyFile () {
  device=$1
  filename=$2
  classId=$3
  childClassId=$4
  if [ -z "$filename" ] || [ -z "$classId" ];then
    echo "filename and classid paramters required"
  else
    # 只在第一次进入时初始化
    if [ -z "$latestLoss" ]; then
      latestLoss="0%"
    fi
    if [ -z "$latestDelay" ]; then
      latestDelay="0ms"
    fi
    while read -r line; do
      if [[ $line == \#* ]];then
        continue;
      else
        keys=($line)
        comm=${keys[0]}
        value=${keys[1]}
        case $comm in
          rate)
            echo "setting rate on $device $classId $value"
            $TC class change dev $device parent 1: classid 1:$classId htb rate $value
            $TC class change dev $device parent 1: classid 1:$childClassId htb rate $value
            ;;
          loss)
            latestLoss=$value
            echo "setting loss on $device $childClassId $latestLoss"
            $TC qdisc replace dev $device parent 1:$childClassId handle 10: netem loss $latestLoss delay $latestDelay
            ;;
          delay)
            latestDelay=$value
            echo "setting delay on $device $childClassId $latestDelay"
            $TC qdisc replace dev $device parent 1:$childClassId handle 10: netem loss $latestLoss delay $latestDelay
            ;;
          wait)
            echo "waiting for $device $value seconds"
            sleep $value
            ;;
        esac
      fi
    done < "$filename"
  fi
}

policyLoop () {
  device=$1
  filename=$2
  classId=$3
  childClassId=$4
  while true; do
    parsePolicyFile $device $filename $classId $childClassId
  done
}

currentIfNo=1
while [[ -v INTERFACE_$currentIfNo ]]; do
  interface_var="INTERFACE_$currentIfNo"
  interface="${!interface_var}"
  # 【改动】在清理旧规则时增加容错处理。
  # `2>/dev/null || true` 的意思是：如果命令执行失败（比如接口上本来就没有qdisc），就忽略错误并继续执行，防止脚本意外退出。
  $TC qdisc del dev $interface root 2>/dev/null || true
  $TC qdisc add dev $interface root handle 1: htb default 10
  ((currentIfNo++))
done

currentIfNo=1
while [[ -v PORT_$currentIfNo ]]; do
  interface_var="INTERFACE_$currentIfNo"
  interface="${!interface_var}"
  port_var="PORT_$currentIfNo"
  port="${!port_var}"
  file_var="FILE_$currentIfNo"
  file="${!file_var}"

  childIfNo=${currentIfNo}0
  $TC class add dev $interface parent 1: classid 1:$currentIfNo htb rate 1024Mbps
  $TC class add dev $interface parent 1:$currentIfNo classid 1:$childIfNo htb rate 1024Mbps

  # 【改动】移除 sfq，因为我们将在 policyLoop 中动态添加/替换 netem。
  # 这样避免了 sfq 和 netem 的逻辑冲突，让代码更清晰。
  # 原命令: $TC qdisc add dev $interface parent 1:$childIfNo handle 10: sfq perturb 10

  # 【最核心的改动】将过滤规则从 sport 改为 dport。
  # 因为你是在发送端对出站流量进行限制，发往接收端 8000 端口的数据包，其“目的端口(dport)”才是 8000。
  # “源端口(sport)”是发送端系统随机分配的，所以原来的规则无法匹配到你的音视频流。
  $TC filter add dev $interface parent 1:0 protocol ip prio 1 u32 match ip dport $port 0xffff flowid 1:$childIfNo
  policyLoop $interface $file $currentIfNo $childIfNo &
  ((currentIfNo++))
done

wait