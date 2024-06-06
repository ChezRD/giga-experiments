from ipex_llm import optimize_model
from sage.spelling_correction import AvailableCorrectors
from sage.spelling_correction import T5ModelForSpellingCorruption

spelling_model = T5ModelForSpellingCorruption.from_pretrained(AvailableCorrectors.sage_mt5_large.value)

model = optimize_model(spelling_model.model) # With only one line code change
# Use the optimized model without other API change
# model.batch_correct(['йоу'])

inputs = spelling_model.tokenizer('тестовая строка', return_tensors="pt", padding="longest", truncation=False, max_length=1024)
generated_tokens = model.generate(**inputs.to(model.device), max_length = inputs["input_ids"].size(1) * 1.5)
result = spelling_model.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(result)

# model.generate('йоу', max_length=50)
# (Optional) you can also save the optimized model by calling 'save_low_bit'
model.save_low_bit("{AvailableCorrectors.sage_mt5_large.value}_low_bit")

# # from pyannote.audio import Pipeline
# # pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",use_auth_token="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")
# # output = pipeline("./downloads/nts.wav.wav")

# import os
# import logging
# logging.basicConfig(level = logging.INFO)
# logging.disable(logging.CRITICAL)
# import nemo.collections.asr

# import torch
# from tqdm import tqdm
# import numpy as np
# from typing import List, Tuple
# try:
#     import intel_extension_for_pytorch
# except ImportError:
#     print("xpu not available")


# from nemo.collections.asr.models import EncDecCTCModel
# from nemo.collections.asr.modules.audio_preprocessing import (
#     AudioToMelSpectrogramPreprocessor as NeMoAudioToMelSpectrogramPreprocessor,
# )
# from nemo.collections.asr.parts.preprocessing.features import (
#     FilterbankFeaturesTA as NeMoFilterbankFeaturesTA,
# )
# import torchaudio

# # audio_path = "./downloads/nts.wav.wav"
# # audio_path = "./data/long_example.wav"
# audio_path = "./data/example.wav"
# model_config = "./data/ctc_model_config.yaml"
# model_weights = "./data/ctc_model_weights.ckpt"
# device = "xpu"
# fp16 = False
# batch_size = 16
# sampling_rate = 16000

# def tensor_segment_to_numpy(tensor: torch.Tensor) -> np.ndarray:
#     """Convert a segment of torch.Tensor to numpy array."""
#     samples = tensor.detach().numpy()
    
#     samples = samples.astype(np.float32, order="C") / 32768.0
#     return samples

# def segment_audio(
#     audio_path: str,
#     max_duration: float = 22.0,
#     min_duration: float = 15.0,
#     new_chunk_threshold: float = 0.2,
# ) -> Tuple[List[np.ndarray], List[List[float]]]:
#     print("Loading Silero VAD...")
#     model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, verbose=False)
#     (get_speech_timestamps, _, read_audio, _, _) = utils

#     print("Loading audio...")
#     audio = read_audio(audio_path, sampling_rate=sampling_rate)
        
#     timestamps = []

#     with tqdm(total=100, desc="Analyzing audio for voice fragments") as pbar:
#         def tqdm_callback(pointer):
#             progress = int(pointer)
#             pbar.update(progress - pbar.n)

#         timestamps = get_speech_timestamps(
#             audio,
#             model,
#             sampling_rate=sampling_rate,
#             progress_tracking_callback=tqdm_callback
#         )

#     segments = []
#     curr_duration = 0
#     curr_start = 0
#     curr_end = 0
#     boundaries = []

#     for time_part in tqdm(timestamps, desc="Splitting audio into segments"):
#         start = max(0, int(time_part["start"]))
#         end = min(int(len(audio)), int(time_part["end"]))
        
#         if (
#             curr_duration > min_duration and start - curr_end > new_chunk_threshold
#         ) or (curr_duration + (end - curr_end) > max_duration):
#             audio_segment = audio[curr_start:curr_end]
#             segments.append(audio_segment)
#             boundaries.append([curr_start / sampling_rate, curr_end / sampling_rate])
#             curr_start = start

#         curr_end = end
#         curr_duration = curr_end - curr_start

#     if curr_duration != 0:
#         audio_segment = audio[curr_start:curr_end]
#         segments.append(audio_segment)
#         boundaries.append([curr_start / sampling_rate, curr_end / sampling_rate])

#     non_empty_segments = []
#     non_empty_boundaries = []
#     for i, (segment, boundary) in enumerate(zip(segments, boundaries)):
#         start, end = boundary
#         if start != end:
#             non_empty_segments.append(segment)
#             non_empty_boundaries.append(boundary)

#     return non_empty_segments, non_empty_boundaries

# segments, boundaries = segment_audio(audio_path)


# class FilterbankFeaturesTA(NeMoFilterbankFeaturesTA):
#     def __init__(self, mel_scale: str = "htk", wkwargs=None, **kwargs):
#         if "window_size" in kwargs:
#             del kwargs["window_size"]
#         if "window_stride" in kwargs:
#             del kwargs["window_stride"]

#         super().__init__(**kwargs)

#         self._mel_spec_extractor = torchaudio.transforms.MelSpectrogram(
#             sample_rate=self._sample_rate,
#             win_length=self.win_length,
#             hop_length=self.hop_length,
#             n_mels=kwargs["nfilt"],
#             window_fn=self.torch_windows[kwargs["window"]],
#             mel_scale=mel_scale,
#             norm=kwargs["mel_norm"],
#             n_fft=kwargs["n_fft"],
#             f_max=kwargs.get("highfreq", None),
#             f_min=kwargs.get("lowfreq", 0),
#             wkwargs=wkwargs,
#         )


# class AudioToMelSpectrogramPreprocessor(NeMoAudioToMelSpectrogramPreprocessor):
#     def __init__(self, mel_scale: str = "htk", **kwargs):
#         super().__init__(**kwargs)
#         kwargs["nfilt"] = kwargs["features"]
#         del kwargs["features"]
#         self.featurizer = (
#             FilterbankFeaturesTA(  # Deprecated arguments; kept for config compatibility
#                 mel_scale=mel_scale,
#                 **kwargs,
#             )
#         )

# def format_time(seconds: float) -> str:
#     hours = int(seconds // 3600)
#     minutes = int((seconds % 3600) // 60)
#     seconds = seconds % 60
#     full_seconds = int(seconds)
#     milliseconds = int((seconds - full_seconds) * 100)

#     if hours > 0:
#         return f"{hours:02}:{minutes:02}:{full_seconds:02}:{milliseconds:02}"
#     else:
#         return f"{minutes:02}:{full_seconds:02}:{milliseconds:02}"
    

# model = EncDecCTCModel.from_config_file(model_config)
# logging.basicConfig(level = logging.INFO)
# logging.getLogger("nemo_logger").setLevel(logging.FATAL)

# ckpt = torch.load(model_weights, map_location="cpu")
# model.load_state_dict(ckpt, strict=False)
# model = model.to(device)
# model.eval()

# # Transcribe segments
# transcriptions = []
# transcriptions = model.transcribe(segments, batch_size=int(batch_size))

# for transcription, boundary in zip(transcriptions, boundaries):
#     print(
#         f"[{format_time(boundary[0])} - {format_time(boundary[1])}]: {transcription}\n"
#     )