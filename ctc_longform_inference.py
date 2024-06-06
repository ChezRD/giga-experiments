import logging
logging.basicConfig(level = logging.INFO)
logging.disable(logging.CRITICAL)
import torch
import nemo.collections.asr
import argparse
import json
import math
# from rupunkt import RUPunkt
from typing import List, Tuple
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    print("xpu not available")
    if torch.cuda.is_available():
        torch.cuda.memory.change_current_allocator( torch.cuda.memory.CUDAPluggableAllocator('/usr/local/lib/alloc.so','gtt_alloc','gtt_free') )

import numpy as np
from tqdm import tqdm
import torchaudio
from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessor as NeMoAudioToMelSpectrogramPreprocessor,
)
from nemo.collections.asr.parts.preprocessing.features import (
    FilterbankFeaturesTA as NeMoFilterbankFeaturesTA,
)

class FilterbankFeaturesTA(NeMoFilterbankFeaturesTA):
    def __init__(self, mel_scale: str = "htk", wkwargs=None, **kwargs):
        if "window_size" in kwargs:
            del kwargs["window_size"]
        if "window_stride" in kwargs:
            del kwargs["window_stride"]

        super().__init__(**kwargs)

        self._mel_spec_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=self._sample_rate,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=kwargs["nfilt"],
            window_fn=self.torch_windows[kwargs["window"]],
            mel_scale=mel_scale,
            norm=kwargs["mel_norm"],
            n_fft=kwargs["n_fft"],
            f_max=kwargs.get("highfreq", None),
            f_min=kwargs.get("lowfreq", 0),
            wkwargs=wkwargs,
        )

class AudioToMelSpectrogramPreprocessor(NeMoAudioToMelSpectrogramPreprocessor):
    def __init__(self, mel_scale: str = "htk", **kwargs):
        super().__init__(**kwargs)
        kwargs["nfilt"] = kwargs["features"]
        del kwargs["features"]
        self.featurizer = (
            FilterbankFeaturesTA(  # Deprecated arguments; kept for config compatibility
                mel_scale=mel_scale,
                **kwargs,
            )
        )

class NpJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpJsonEncoder, self).default(obj)

def add_silence_line_breaks(transcriptions: List[str], boundaries: List[List[float]]) -> List[str]:
    # Calculate silence threshold
    silence_durations = [boundaries[i+1][0] - boundaries[i][1] for i in range(len(boundaries) - 1)]
    median_silence = np.median(silence_durations)
    HARD_SILENCE_THRESHOLD_SECONDS = 2 * median_silence  # Adjust as needed
    SOFT_SILENCE_THRESHOLD_SECONDS = 1.5 * median_silence  # Adjust as needed

    # Add line breaks based on calculated silence threshold
    transcriptions_with_line_breaks = []
    for i, segment in enumerate(transcriptions):
        if i > 0:
            silence_duration = boundaries[i][0] - boundaries[i-1][1]
            if silence_duration > HARD_SILENCE_THRESHOLD_SECONDS:
                transcriptions_with_line_breaks.append("\n\n")
            elif silence_duration > SOFT_SILENCE_THRESHOLD_SECONDS:
                transcriptions_with_line_breaks.append("\n")
        transcriptions_with_line_breaks.append(segment)

    return transcriptions_with_line_breaks

def cast_types(value, types_map):
    """
    recurse into value and cast any np.int64 to int

    fix: TypeError: Object of type int64 is not JSON serializable

    import numpy as np
    import json
    data = [np.int64(123)]
    data = cast_types(data, [
        (np.int64, int),
        (np.float64, float),
    ])
    data_json = json.dumps(data)
    data_json == "[123]"

    https://stackoverflow.com/a/75552723/10440128
    """
    if isinstance(value, dict):
        # cast types of dict keys and values
        return {cast_types(k, types_map): cast_types(v, types_map) for k, v in value.items()}
    if isinstance(value, list):
        # cast types of list values
        return [cast_types(v, types_map) for v in value]
    for f, t in types_map:
        if isinstance(value, f):
            return t(value) # cast type of value
    return value # keep type of value

def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    full_seconds = int(seconds)
    milliseconds = int((seconds - full_seconds) * 100)

    if hours > 0:
        return f"{hours:02}:{minutes:02}:{full_seconds:02}:{milliseconds:02}"
    else:
        return f"{minutes:02}:{full_seconds:02}:{milliseconds:02}"

def tensor_segment_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a segment of torch.Tensor to numpy array."""
    samples = tensor.detach().numpy()
    
    samples = samples.astype(np.float32, order="C") / 32768.0
    return samples

def segment_audio(
    audio_path: str,
    max_duration: float = 25.0,
    min_duration: float = 13.0,
    new_chunk_threshold: float = 0.2,
    sampling_rate: int = 16000
) -> Tuple[List[np.ndarray], List[List[float]]]:
    
    pbar = tqdm(total=1, desc="Loading Silero VAD...", bar_format="{desc}")

    try:
        # Load the Silero VAD model from the PyTorch Hub
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', 
            model='silero_vad', 
            force_reload=False, 
            verbose=False
        )
        
        # Unpack the utility functions provided by the model
        get_speech_timestamps, _, read_audio, _, _ = utils
        
        # Update the progress bar to indicate loading is complete
        pbar.set_description_str("Silero VAD loaded.")
        pbar.update(1)
    except Exception as e:
        pbar.set_postfix_str("Error: {e}")
    finally:
        pbar.close()

    pbar = tqdm(total=1, desc="Loading audio...", bar_format="{desc}")
    try:
        audio = read_audio(audio_path, sampling_rate=sampling_rate)
        pbar.set_description_str("Audio loaded.")
        pbar.update(1)
    except Exception as e:
        pbar.set_postfix_str("Error: {e}")
    finally:
        pbar.close()

    timestamps = []

    with tqdm(total=100, desc="Analyzing audio for voice fragments") as pbar:
        def tqdm_callback(pointer):
            progress = int(pointer)
            pbar.update(progress - pbar.n)

        timestamps = get_speech_timestamps(
            audio,
            model,
            threshold=0.6,
            min_speech_duration_ms=300,
            min_silence_duration_ms=400,
            sampling_rate=sampling_rate,
            progress_tracking_callback=tqdm_callback
        )

    segments = []
    curr_duration = 0
    curr_start = 0
    curr_end = 0
    boundaries = []

    for time_part in tqdm(timestamps, desc="Splitting audio into segments"):
        start = max(0, int(time_part["start"]))
        end = min(int(len(audio)), int(time_part["end"]))
        
        if (
            curr_duration > min_duration and start - curr_end > new_chunk_threshold
        ) or (curr_duration + (end - curr_end) > max_duration):
            audio_segment = audio[curr_start:curr_end]
            segments.append(audio_segment)
            boundaries.append([curr_start / sampling_rate, curr_end / sampling_rate])
            curr_start = start

        curr_end = end
        curr_duration = curr_end - curr_start

    if curr_duration != 0:
        audio_segment = audio[curr_start:curr_end]
        segments.append(audio_segment)
        boundaries.append([curr_start / sampling_rate, curr_end / sampling_rate])

    non_empty_segments = []
    non_empty_boundaries = []
    for i, (segment, boundary) in enumerate(zip(segments, boundaries)):
        start, end = boundary
        if start != end:
            non_empty_segments.append(segment)
            non_empty_boundaries.append(boundary)

    print("Unloading Silero VAD...")
    # model.detach()
    model.reset_states()
    del get_speech_timestamps, read_audio, model
    print("Silero VAD unloaded.")

    return non_empty_segments, non_empty_boundaries

def process_transcriptions(transcriptions):
    result = []
    current_group = []
    for i, s in enumerate(transcriptions):
        if s == "\n\n":
            if current_group:
                result.append(" ".join(current_group).split("\n"))
            current_group = []
        else:
            current_group.append(s)
    if current_group:
        result.append(" ".join(current_group).split("\n"))
    return result

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run long-form inference using GigaAM-CTC checkpoint"
    )
    parser.add_argument("--model_config", help="Path to GigaAM-CTC config file (.yaml)")
    parser.add_argument(
        "--model_weights", help="Path to GigaAM-CTC checkpoint file (.ckpt)"
    )
    parser.add_argument("--audio_path", help="Path to audio signal")
    parser.add_argument("--device", help="Device: cpu / cuda")
    parser.add_argument("--fp16", help="Run in FP16 mode", default=True)
    parser.add_argument(
        "--batch_size", help="Batch size for acoustic model inference", default=10
    )
    return parser.parse_args()

def main(
    model_config: str,
    model_weights: str,
    device: str,
    audio_path: str,
    # hf_token: str,
    fp16: bool,
    batch_size: int = 10,
    transcript_path: str = 'output.json'
):
    batch_size = int(batch_size)
    segments, boundaries = segment_audio(audio_path)
    # segments_with_line_breaks = add_silence_line_breaks(segments, boundaries)

    print("Loading CTC model...")    
    model = EncDecCTCModel.from_config_file(model_config)
    logging.basicConfig(level = logging.INFO)
    logging.getLogger("nemo_logger").setLevel(logging.FATAL)

    ckpt = torch.load(model_weights, map_location=torch.device(device))
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device)
    if device != "cpu" and fp16:
        model = model.half()
        model.preprocessor = model.preprocessor.float()
    model.eval()
    print("CTC model loaded.")

    print("Begin transcriptions...")
    # Transcribe segments
    transcriptions = []
    if device != "cpu" and fp16:
        with torch.autocast(device_type=device, dtype=torch.float16):
            transcriptions = model.transcribe(segments, batch_size=batch_size)
    else:
        transcriptions = model.transcribe(segments, batch_size=batch_size)

    transcriptions = add_silence_line_breaks(transcriptions, boundaries)

    print("Transcriptions ended.")

    print("Unloading CTC model...")
    # del model.model
    # model.cpu()
    del model
    print("CTC model unloaded.")

    print("Sage model loading...")
    from sage.spelling_correction import AvailableCorrectors
    from sage.spelling_correction import T5ModelForSpellingCorruption
    spelling_model = T5ModelForSpellingCorruption.from_pretrained(AvailableCorrectors.sage_mt5_large.value)
    
    spelling_model.model.to(torch.device(device))
    if device != "cpu" and fp16:
        spelling_model.model = spelling_model.model.half()
        # spelling_model.preprocessor = spelling_model.preprocessor.float()
    print("Sage model loaded.")

    export_segments = []
    for boundary, transcription in zip(boundaries, transcriptions):
        stripped = transcription.strip()
        if len(stripped) > 0:
            export_segments.append({
                "start": format_time(boundary[0]),
                "end": format_time(boundary[1]),
                "text": transcription,
            })

    def flatten_list_of_lists(array_of_arrays: List[List[str]]) -> List[str]:
        return [item for sublist in array_of_arrays for item in sublist]

    proceeded_transcriptions = process_transcriptions(transcriptions)

    flattened_list = flatten_list_of_lists(proceeded_transcriptions)

    if device != "cpu" and fp16:
        with torch.autocast(device_type=device, dtype=torch.float16):
            corrected_strings = spelling_model.batch_correct(flattened_list, batch_size=batch_size)
    else:
        corrected_strings = spelling_model.batch_correct(flattened_list, batch_size=batch_size)

    flatten_result = "\n\n".join(["\n".join(sub_str_list) for sub_str_list in corrected_strings])

    with open(transcript_path, "w", encoding="utf8") as fp:
        result = {
            "segments": export_segments,
            "text": flatten_result,
        }
        # Your code ... 
        json.dump(cast_types(result, [ (np.int64, int), (np.float64, float) ]), fp, ensure_ascii=False, cls=NpJsonEncoder)

    print(f"Voila!✨ Your file has been transcribed. Check it out here 👉 {transcript_path}")
        


if __name__ == "__main__":
    args = _parse_args()
    main(
        model_config=args.model_config,
        model_weights=args.model_weights,
        device=args.device,
        audio_path=args.audio_path,
        fp16=args.fp16,
        batch_size=args.batch_size,
    )