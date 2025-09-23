#!/bin/bash
TC="tc"
INTERFACE_1=enp4s0
PORT_1=8000

# 【新增】定义要使用的 ifb 虚拟网卡的名字
IFB_DEV="ifb0"

FILE_1=$1
if [ -z "$FILE_1" ]; then
  echo "policy file name has to be specified"
  exit 1;
fi

# 【新增】函数：确保 ifb0 虚拟网卡存在并已激活
# (此函数无需改动)
ensure_ifb() {
  modprobe ifb numifbs=1 2>/dev/null || true
  ip link show $IFB_DEV >/dev/null 2>&1 || ip link add $IFB_DEV type ifb
  ip link set $IFB_DEV up
}


parsePolicyFile () {
  device=$1
  filename=$2
  classId=$3
  childClassId=$4
  if [ -z "$filename" ] || [ -z "$classId" ];then
    echo "filename and classid paramters required"
  else
    # 【修复】将 latestLoss/latestDelay 的初始化放在循环外，以正确保持状态
    latestLoss="0%";
    latestDelay="0ms";
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
        # 【修复】burst 值计算错误，kbit/s 转 byte/s 是除以8，不是800。且单位应为 bytes。
        # 一个更简单的做法是直接让 tc 处理单位
        $TC class change dev $device parent 1: classid 1:$classId htb rate $value
        $TC class change dev $device parent 1: classid 1:$childClassId htb rate $value
        ;;
    loss)
         latestLoss=$value;
        echo "setting loss on $device $childClassId $value"
        $TC qdisc replace dev $device parent 1:$childClassId handle 10: netem loss $latestLoss delay $latestDelay
        ;;
      delay)
        latestDelay=$value;
        echo "setting delay on $device $childClassId $value"
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

# --- 主要逻辑开始 ---

# 确保 ifb0 已经准备就绪
ensure_ifb

currentIfNo=1
while [[ -v INTERFACE_$currentIfNo ]]; do
  interface_var="INTERFACE_$currentIfNo"
  interface="${!interface_var}"

  # 【改动】更彻底地清理旧规则
  $TC qdisc del dev $interface root 2>/dev/null || true
  $TC qdisc del dev $interface clsact 2>/dev/null || true # 清理新的 clsact 队列
  $TC qdisc del dev $IFB_DEV root 2>/dev/null || true

  # 【改动】使用 clsact 代替 ingress，这是更现代和健壮的方式
  # clsact 专门用于在 ingress 和 egress 挂载过滤器，能有效避免 "Invalid argument" 错误
  $TC qdisc add dev $interface clsact

  # 【改动】在 ifb 网卡上创建 htb 队列结构，使用 replace 避免 "Exclusivity flag" 错误
  # 同时添加 r2q 参数可以消除 quantum 警告
  $TC qdisc replace dev $IFB_DEV root handle 1: htb default 10 r2q 10

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

  # 【改动】在 ifb 设备上创建 htb 类别
  $TC class replace dev $IFB_DEV parent 1: classid 1:$currentIfNo htb rate 1024Mbps
  $TC class replace dev $IFB_DEV parent 1:1 classid 1:$childIfNo htb rate 1024Mbps

  # 【改动】修改过滤器，使其挂载在 clsact 的 ingress 钩子上
  # parent ffff: 被替换为 parent ingress，这是 clsact 的标准用法
  # 这个过滤器将物理网卡的入站流量重定向到 ifb 设备
  $TC filter add dev $interface parent ingress protocol ip prio 1 u32 match ip dport $port 0xffff action mirred egress redirect dev $IFB_DEV

  # 【改动】让策略循环在 ifb 设备上运行
  policyLoop $IFB_DEV $file $currentIfNo $childIfNo &
  ((currentIfNo++))
done

wait