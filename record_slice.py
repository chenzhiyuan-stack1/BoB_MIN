import wave
import os

def get_wav_duration(wav_path):
    with wave.open(wav_path, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / float(rate)
    return duration  # 单位：秒

def get_yuv_frame_count(yuv_path, width, height, pixfmt='yuv420p'):
    # yuv420p: 每帧大小 = width * height * 3 / 2 字节
    frame_size = width * height * 3 // 2
    file_size = os.path.getsize(yuv_path)
    return file_size // frame_size