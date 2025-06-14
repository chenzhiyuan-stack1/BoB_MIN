import os
import glob
from skimage import io
from skimage.metrics import niqe
from pydub import AudioSegment
import numpy as np
import soundfile as sf
import librosa
import onnxruntime as ort

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01

def eval_audio_mos_dnsmos(audio_file):
    """
    用DNSMOS本地模型评估音频MOS，返回MOS分数（OVRL），范围约[1,5]，5为最好。
    需提前准备好模型文件。
    """
    class ComputeScore:
        def __init__(self, primary_model_path, p808_model_path) -> None:
            self.onnx_sess = ort.InferenceSession(primary_model_path)
            self.p808_onnx_sess = ort.InferenceSession(p808_model_path)
            
        def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size+1, hop_length=hop_length, n_mels=n_mels)
            if to_db:
                mel_spec = (librosa.power_to_db(mel_spec, ref=np.max)+40)/40
            return mel_spec.T
        
        def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
            if is_personalized_MOS:
                p_ovr = np.poly1d([-0.00533021,  0.005101  ,  1.18058466, -0.11236046])
                p_sig = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
                p_bak = np.poly1d([-0.04976499,  0.44276479, -0.1644611 ,  0.96883132])
            else:
                p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
                p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
                p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])

            sig_poly = p_sig(sig)
            bak_poly = p_bak(bak)
            ovr_poly = p_ovr(ovr)

            return sig_poly, bak_poly, ovr_poly
        
        def __call__(self, fpath, sampling_rate=SAMPLING_RATE, is_personalized_MOS=False):
            aud, input_fs = sf.read(fpath)
            fs = sampling_rate
            if input_fs != fs:
                audio = librosa.resample(aud, input_fs, fs)
            else:
                audio = aud
            actual_audio_len = len(audio)
            len_samples = int(INPUT_LENGTH*fs)
            while len(audio) < len_samples:
                audio = np.append(audio, audio)
            
            num_hops = int(np.floor(len(audio)/fs) - INPUT_LENGTH)+1
            hop_len_samples = fs
            predicted_mos_sig_seg_raw = []
            predicted_mos_bak_seg_raw = []
            predicted_mos_ovr_seg_raw = []
            predicted_mos_sig_seg = []
            predicted_mos_bak_seg = []
            predicted_mos_ovr_seg = []
            predicted_p808_mos = []
            
            for idx in range(num_hops):
                audio_seg = audio[int(idx*hop_len_samples) : int((idx+INPUT_LENGTH)*hop_len_samples)]
                if len(audio_seg) < len_samples:
                    continue

                input_features = np.array(audio_seg).astype('float32')[np.newaxis,:]
                p808_input_features = np.array(self.audio_melspec(audio=audio_seg[:-160])).astype('float32')[np.newaxis, :, :]
                oi = {'input_1': input_features}
                p808_oi = {'input_1': p808_input_features}
                p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
                mos_sig_raw,mos_bak_raw,mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
                mos_sig,mos_bak,mos_ovr = self.get_polyfit_val(mos_sig_raw,mos_bak_raw,mos_ovr_raw,is_personalized_MOS)
                predicted_mos_sig_seg_raw.append(mos_sig_raw)
                predicted_mos_bak_seg_raw.append(mos_bak_raw)
                predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
                predicted_mos_sig_seg.append(mos_sig)
                predicted_mos_bak_seg.append(mos_bak)
                predicted_mos_ovr_seg.append(mos_ovr)
                predicted_p808_mos.append(p808_mos)

            clip_dict = {'filename': fpath, 'len_in_sec': actual_audio_len/fs, 'sr':fs}
            clip_dict['num_hops'] = num_hops
            clip_dict['OVRL_raw'] = np.mean(predicted_mos_ovr_seg_raw)
            clip_dict['SIG_raw'] = np.mean(predicted_mos_sig_seg_raw)
            clip_dict['BAK_raw'] = np.mean(predicted_mos_bak_seg_raw)
            clip_dict['OVRL'] = np.mean(predicted_mos_ovr_seg)
            clip_dict['SIG'] = np.mean(predicted_mos_sig_seg)
            clip_dict['BAK'] = np.mean(predicted_mos_bak_seg)
            clip_dict['P808_MOS'] = np.mean(predicted_p808_mos)
            return clip_dict
    primary_model_path = '../DNS-Challenge/DNSMOS/DNSMOS/sig_bak_ovr.onnx'
    p808_model_path = '../DNS-Challenge/DNSMOS/DNSMOS/model_v8.onnx'
    personalized = False  # 根据需要设置是否使用个性化MOS模型
    scorer = ComputeScore(primary_model_path, p808_model_path)
    result = scorer(audio_file, sampling_rate=SAMPLING_RATE, is_personalized_MOS=personalized)
    return result

def eval_video_mos_niqe(frames_dir):
    """
    用NIQE（reference-free）评估视频MOS，对每帧图片评估NIQE分数，归一化为[0,5]，5为最好。
    需安装scikit-image。
    """
    frame_files = sorted(glob.glob(os.path.join(frames_dir, '*.png')))
    if not frame_files:
        print("未找到帧图片")
        return None
    scores = []
    for f in frame_files:
        img = io.imread(f)
        if img.ndim == 2:  # 灰度
            pass
        elif img.shape[2] == 4:  # RGBA
            img = img[:, :, :3]
        score = niqe(img)
        scores.append(score)
    avg_niqe = np.mean(scores)
    # NIQE分数越低越好，经验上可用如下线性映射到MOS[1,5]（你可根据实际情况调整）
    # 0分对应5分，20分对应1分
    mos = max(1.0, min(5.0, 5 - (avg_niqe / 5)))
    return mos

# 用法示例
if __name__ == "__main__":
    # 评估音频
    result = eval_audio_mos_dnsmos("mi0.wav")
    print("DNSMOS评估结果:", result)
    print("MOS分数（OVRL）:", result['OVRL'])

    # 评估视频（假设mi0_frames目录下是该MI的所有帧图片）
    mos_video = eval_video_mos_niqe("mi0_frames")
    print("视频MOS:", mos_video)