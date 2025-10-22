#!/bin/bash

# 检查是否提供了ID参数
if [ -z "$1" ]; then
    echo "错误: 请提供一个ID作为参数。"
    echo "用法: ./clean.sh <ID>"
    exit 1
fi

ID=$1
BASE_DIR="./results/$ID"

# 检查目标ID目录是否存在
if [ ! -d "$BASE_DIR" ]; then
    echo "错误: 目录 $BASE_DIR 不存在。"
    exit 1
fi

echo "处理 ID 目录: $BASE_DIR"

# 设置 nullglob 选项：如果没有匹配的目录，循环将不执行
# 而不是返回一个包含通配符的字符串
shopt -s nullglob

# 遍历 BASE_DIR 下的所有子目录
# "*/" 确保我们只匹配目录
for dir in "$BASE_DIR"/*/; do
    
    # 检查 data.jsonl 文件是否存在于该子目录中
    # ${dir} 变量已包含末尾的斜杠, e.g., ./results/19/subdir/
    if [ -f "${dir}data.jsonl" ]; then
        # 文件存在，保留
        echo "  保留: $dir (存在 data.jsonl)"
    else
        # 文件不存在，删除
        echo "  已删除: $dir"
        # 使用 rm -rf 递归并强制删除目录及其所有内容
        rm -rf "$dir"
    fi
done

# 恢复默认的 glob 行为
shopt -u nullglob

echo "清理完成。"