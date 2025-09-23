#!/bin/bash
TC='tc'
INTERFACE1=enp0s31f6

killall tc_policy_min3.sh 1>/dev/null 2>&1
killall sleep 1>/dev/null 2>&1
killall tc 1>/dev/null 2>&1

$TC qdisc del dev $INTERFACE1 root handle 1:0 1>/dev/null 2>&1
$TC qdisc del dev $INTERFACE1 root 1>/dev/null 2>&1
$TC qdisc del dev lo root 1>/dev/null 2>&1
