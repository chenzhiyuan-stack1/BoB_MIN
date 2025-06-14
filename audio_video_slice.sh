# This script processes audio and video files based on a provided text file.
# It splits the audio into segments and extracts video frames for each segment.
# duration,frame_count
# 2.0,60
# 2.0,120
# 2.0,180
docker run --rm -v `pwd`:/app -w /app challenge-env bash -c '
# 1. 先转出所有视频帧图片
ffmpeg -s 640x360 -pix_fmt yuv420p -i outvideo.yuv -vf fps=30 outvideo_frame_%06d.png

# 2. 读取mi_info.txt，分割音频和视频
audio_start=0
video_start=1
idx=0
while IFS=, read -r duration frame_end
do
  # 跳过表头
  if [[ $duration == duration* ]]; then continue; fi

  # 音频分割
  audio_end=$(echo "$audio_start + $duration" | bc)
  ffmpeg -y -i outaudio.wav -ss $audio_start -to $audio_end -c copy mi_audio_${idx}.wav

  # 视频分割
  mkdir -p mi${idx}_frames
  for ((i=video_start; i<=frame_end; i++)); do
    printf -v frame_name "outvideo_frame_%06d.png" "$i"
    cp "$frame_name" mi${idx}_frames/
  done

  # 更新下一个MI的起点
  audio_start=$audio_end
  video_start=$((frame_end+1))
  idx=$((idx+1))
done < mi_info.txt
'