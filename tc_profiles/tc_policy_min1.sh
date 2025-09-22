#!/bin/bash

# tc 命令路径，通常在 /sbin/tc 或 /usr/sbin/tc
TC="tc"

# ==============================================================================
# 配置区域
# ==============================================================================
# 定义要进行流量控制的网络接口。
# 根据你的 tcpdump 输出，这里应该是 enp4s0。
INTERFACE_1=enp4s0

# 策略文件从第一个命令行参数获取
FILE_1=$1

# ==============================================================================

# 检查是否提供了策略文件
if [ -z "$FILE_1" ]; then
  echo "错误: 必须指定一个策略文件名作为参数。"
  echo "用法: $0 <策略文件名>"
  exit 1
fi

#
# 函数: parsePolicyFile
# 作用: 读取并解析策略文件，应用 tc 命令
#
parsePolicyFile () {
  local device=$1
  local filename=$2
  local classId=$3 # 这是HTB的默认类别ID，例如 10

  # 如果文件不存在，则退出
  if [ ! -f "$filename" ]; then
    echo "错误: 策略文件 '$filename' 未找到。"
    # 让循环继续，但打印错误
    sleep 5
    return
  fi

  # 初始化/重置 netem 参数
  local latestLoss="0%"
  local latestDelay="0ms"

  # 循环读取策略文件的每一行
  while read -r line; do
    # 跳过空行和注释行
    if [[ -z "$line" ]] || [[ $line == \#* ]]; then
      continue
    fi

    # 解析命令和值
    keys=($line)
    comm=${keys[0]}
    value=${keys[1]}

    case $comm in
      rate)
        echo "设置速率: $device, 类别 1:$classId, 速率 $value"
        # 计算合理的 burst 值
        burst=$(echo "$value" | awk -F'kbit' '{print $1 * 1.5}')
        burst="${burst}k"
        $TC class change dev "$device" parent 1: classid 1:"$classId" htb rate "$value" burst "$burst"
        ;;
      loss)
        latestLoss=$value
        echo "设置丢包/延迟: $device, 类别 1:$classId, 丢包 $latestLoss, 延迟 $latestDelay"
        # 尝试修改现有的 netem 规则，如果失败（说明 netem 不存在），则添加一个新的 netem 规则
        $TC qdisc change dev "$device" parent 1:"$classId" netem loss "$latestLoss" delay "$latestDelay" \
        || $TC qdisc add dev "$device" parent 1:"$classId" netem loss "$latestLoss" delay "$latestDelay"
        ;;
      delay)
        latestDelay=$value
        echo "设置丢包/延迟: $device, 类别 1:$classId, 丢包 $latestLoss, 延迟 $latestDelay"
        # 逻辑同上
        $TC qdisc change dev "$device" parent 1:"$classId" netem loss "$latestLoss" delay "$latestDelay" \
        || $TC qdisc add dev "$device" parent 1:"$classId" netem loss "$latestLoss" delay "$latestDelay"
        ;;
      wait)
        echo "等待: $value"
        sleep "$value"
        ;;
      *)
        echo "警告: 未知的命令 '$comm' 在文件 '$filename' 中"
        ;;
    esac
  done < "$filename"
}

#
# 函数: policyLoop
# 作用: 无限循环执行策略文件解析
#
policyLoop () {
  local device=$1
  local filename=$2
  local classId=$3
  while true; do
    parsePolicyFile "$device" "$filename" "$classId"
  done
}

# ==============================================================================
# 主执行逻辑
# ==============================================================================

echo "--- 初始化 TC 队列结构 ---"

# 遍历所有定义的 INTERFACE_x 变量
currentIfNo=1
while [[ -v INTERFACE_$currentIfNo ]]; do
  # 获取真实的网卡名
  interface_var="INTERFACE_$currentIfNo"
  interface="${!interface_var}"
  
  echo "正在初始化接口: $interface"

  # 1. 清理该接口上所有旧的 tc 规则，确保环境干净
  $TC qdisc del dev "$interface" root

  # 2. 添加一个 HTB 根队列，句柄为 1:
  #    关键: `default 10` 表示所有未被过滤器匹配的流量都将进入类别 1:10
  $TC qdisc add dev "$interface" root handle 1: htb default 10

  # 3. 创建这个默认的类别 1:10，并给一个初始的高速率
  $TC class add dev "$interface" parent 1: classid 1:10 htb rate 1024Mbps

  # 4. 获取对应的策略文件
  file_var="FILE_$currentIfNo"
  file="${!file_var}"

  # 5. 在后台启动策略循环，对默认类别 10 进行操作
  if [ -n "$file" ]; then
    echo "为接口 $interface 启动策略循环，使用文件 $file"
    policyLoop "$interface" "$file" 10 &
  else
    echo "警告: 未为接口 $interface 定义 FILE_$currentIfNo 策略文件。"
  fi

  ((currentIfNo++))
done 

echo "--- TC 策略已启动，脚本将保持运行 ---"

# 等待所有后台进程，使脚本保持活动状态
wait