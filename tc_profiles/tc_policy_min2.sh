#!/bin/bash
set -euo pipefail

TC="tc"
INTERFACE_1=${INTERFACE_1:-enp4s0}
PORT_1=${PORT_1:-8000}
FILE_1=${1:-}

if [[ -z "${FILE_1}" ]]; then
  echo "错误: 必须指定一个策略文件名作为参数。"
  echo "用法: $0 <策略文件名>"
  exit 1
fi

ensure_ifb() {
  modprobe ifb numifbs=1 2>/dev/null || true
  ip link show ifb0 >/dev/null 2>&1 || ip link add ifb0 type ifb
  ip link set ifb0 up
}

setup_ingress_redirect() {
  local iface=$1
  local port=$2

  # 清理旧规则（容错）
  $TC qdisc del dev "$iface" clsact 2>/dev/null || true
  $TC qdisc del dev "$iface" ingress 2>/dev/null || true
  $TC filter del dev "$iface" parent ffff: 2>/dev/null || true

  echo "[tc] 尝试使用 clsact"
  if $TC qdisc add dev "$iface" clsact 2>/dev/null; then
    parent="parent ffff:"
  else
    echo "[tc] clsact 不可用，回退到 ingress"
    $TC qdisc add dev "$iface" handle ffff: ingress
    parent="parent ffff:"
  fi

  # 优先 flower（更稳），失败则回退 u32
  echo "[tc] 尝试 flower 匹配 UDP/TCP dport=$port 重定向到 ifb0"
  if ! $TC filter add dev "$iface" $parent protocol ip prio 10 flower ip_proto udp dst_port "$port" \
        action mirred egress redirect dev ifb0 2>/dev/null; then
    echo "[tc] flower/udp 失败，回退 u32"
    $TC filter add dev "$iface" $parent protocol ip prio 10 u32 \
      match ip protocol 17 0xff \
      match u16 0x$(printf '%04x' "$port") 0xffff at 22 \
      action mirred egress redirect dev ifb0
  fi

  if ! $TC filter add dev "$iface" $parent protocol ip prio 11 flower ip_proto tcp dst_port "$port" \
        action mirred egress redirect dev ifb0 2>/dev/null; then
    echo "[tc] flower/tcp 失败，回退 u32"
    $TC filter add dev "$iface" $parent protocol ip prio 11 u32 \
      match ip protocol 6 0xff \
      match u16 0x$(printf '%04x' "$port") 0xffff at 22 \
      action mirred egress redirect dev ifb0
  fi
}

setup_ifb_shaper() {
  $TC qdisc del dev ifb0 root 2>/dev/null || true
  # 可选: 设置 r2q 减少 HTB 告警
  $TC qdisc add dev ifb0 root handle 1: htb default 10 r2q 10
  $TC class add dev ifb0 parent 1: classid 1:1  htb rate 1024Mbit quantum 1500
  $TC class add dev ifb0 parent 1:1 classid 1:10 htb rate 1024Mbit quantum 1500
  # 预创建 netem
  $TC qdisc add dev ifb0 parent 1:10 handle 10: netem
}

parsePolicyFile () {
  local device_print=$1
  local filename=$2
  local classId=$3
  local shape_dev=$4

  if [[ ! -f "$filename" ]]; then
    echo "错误: 策略文件 '$filename' 未找到。"
    sleep 5
    return
  fi

  local latestLoss="0%"
  local latestDelay="0ms"

  while read -r line; do
    [[ -z "$line" ]] && continue
    [[ "$line" =~ ^# ]] && continue

    read -r comm value <<<"$line"
    case "$comm" in
      rate)
        echo "设置速率: $device_print, 类别 1:$classId, 速率 $value"
        $TC class change dev "$shape_dev" parent 1: classid 1:1  htb rate "$value" quantum 1500 2>/dev/null \
          || $TC class add dev "$shape_dev" parent 1: classid 1:1  htb rate "$value" quantum 1500
        $TC class change dev "$shape_dev" parent 1: classid 1:"$classId" htb rate "$value" quantum 1500 2>/dev/null \
          || $TC class add dev "$shape_dev" parent 1: classid 1:"$classId" htb rate "$value" quantum 1500
        ;;
      loss)
        latestLoss="$value"
        echo "设置丢包/延迟: $device_print, 类别 1:$classId, 丢包 $latestLoss, 延迟 $latestDelay"
        $TC qdisc replace dev "$shape_dev" parent 1:"$classId" handle 10: netem loss "$latestLoss" delay "$latestDelay"
        ;;
      delay)
        latestDelay="$value"
        echo "设置丢包/延迟: $device_print, 类别 1:$classId, 丢包 $latestLoss, 延迟 $latestDelay"
        $TC qdisc replace dev "$shape_dev" parent 1:"$classId" handle 10: netem loss "$latestLoss" delay "$latestDelay"
        ;;
      wait)
        echo "等待: $value"
        sleep "$value"
        ;;
      *)
        echo "警告: 未知命令: $comm $value"
        ;;
    esac
  done < "$filename"
}

policyLoop () {
  local device_print=$1
  local filename=$2
  local classId=$3
  local shape_dev=$4
  while true; do
    parsePolicyFile "$device_print" "$filename" "$classId" "$shape_dev"
  done
}

echo "--- 初始化 TC 队列结构 ---"

currentIfNo=1
while [[ -v INTERFACE_$currentIfNo ]]; do
  interface_var="INTERFACE_$currentIfNo"; interface="${!interface_var}"
  port_var="PORT_$currentIfNo"; port="${!port_var:-$PORT_1}"
  file_var="FILE_$currentIfNo"; file="${!file_var:-$FILE_1}"

  echo "正在初始化接口: $interface (ingress -> ifb0), 端口: $port"

  ensure_ifb
  setup_ingress_redirect "$interface" "$port"
  setup_ifb_shaper

  if [[ -n "$file" ]]; then
    echo "为接口 $interface 启动策略循环，使用文件 $file"
    policyLoop "$interface" "$file" 10 "ifb0" &
  else
    echo "警告: 未为接口 $interface 定义策略文件。"
  fi

  ((currentIfNo++))
done

echo "--- TC 策略已启动，脚本将保持运行 ---"
# 打印一次状态，便于核验
$TC -s qdisc ls dev "$INTERFACE_1" || true
$TC -s filter show dev "$INTERFACE_1" parent ffff: || true
$TC -s qdisc ls dev ifb0 || true

wait