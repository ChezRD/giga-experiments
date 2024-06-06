import logging
logging.basicConfig(level = logging.INFO)
logging.disable(logging.CRITICAL)

import argparse
from typing import List, Tuple, Optional, Union, Any

import json
import numpy as np
import torch
import math
from tqdm import tqdm
import gc  # Import the garbage collector

try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    print("xpu not available")

import torchaudio
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessor as NeMoAudioToMelSpectrogramPreprocessor,
)
from nemo.collections.asr.parts.preprocessing.features import (
    FilterbankFeaturesTA as NeMoFilterbankFeaturesTA,
)
from omegaconf import OmegaConf, open_dict


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
    max_duration: float = 23.0,
    min_duration: float = 12.0,
    new_chunk_threshold: float = 0.3,
    sampling_rate: int = 16000
) -> Tuple[List[np.ndarray], List[List[float]]]:
    
    pbar = tqdm(total=1, desc="Loading Silero VAD...", bar_format="{desc}")

    try:
        # Load the Silero VAD model from the PyTorch Hub
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', 
            model='silero_vad', 
            force_reload=False,
            verbose=False,
            onnx=True,
            force_onnx_cpu=True
        )
       
        # Unpack the utility functions provided by the model
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
        
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
            curr_duration > min_duration and (start - curr_end) / sampling_rate > new_chunk_threshold
        ) or (curr_duration + (end - curr_start) / sampling_rate > max_duration):
            audio_segment = audio[curr_start:curr_end]
            segments.append(audio_segment)
            boundaries.append([curr_start / sampling_rate, curr_end / sampling_rate])
            curr_start = start
            curr_duration = 0

        curr_end = end
        curr_duration = (curr_end - curr_start) / sampling_rate

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
    # Free memory
    del audio
    del segments
    # del non_empty_segments
    # del non_empty_boundaries
    # model.detach()
    model.reset_states()
    del get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks, utils, model
    gc.collect()
    print("Silero VAD unloaded.")

    return non_empty_segments, non_empty_boundaries

# def process_transcriptions(transcriptions):
#     result = []
#     current_group = []
#     for i, s in enumerate(transcriptions):
#         if s == "\n\n":
#             if current_group:
#                 result.append(" ".join(current_group).split("\n"))
#             current_group = []
#         else:
#             current_group.append(s)
#     if current_group:
#         result.append(" ".join(current_group).split("\n"))
#     return result

def process_transcriptions(transcriptions, max_length=512):
    result = []
    current_group = []

    def split_into_chunks(text, max_length):
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 > max_length:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1  # +1 for the space

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    for i, s in enumerate(transcriptions):
        if s == "\n\n":
            if current_group:
                combined_text = " ".join(current_group)
                split_chunks = split_into_chunks(combined_text, max_length)
                for chunk in split_chunks:
                    result.append(chunk.split("\n"))
            current_group = []
        else:
            current_group.append(s)

    if current_group:
        combined_text = " ".join(current_group)
        split_chunks = split_into_chunks(combined_text, max_length)
        for chunk in split_chunks:
            result.append(chunk.split("\n"))

    return result


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run long-form inference using GigaAM-RNNT checkpoint"
    )
    parser.add_argument(
        "--model_config", help="Path to GigaAM-RNNT config file (.yaml)"
    )
    parser.add_argument(
        "--model_weights", help="Path to GigaAM-RNNT checkpoint file (.ckpt)"
    )
    parser.add_argument("--tokenizer_path", help="Path to tokenizer directory")
    parser.add_argument("--audio_path", help="Path to audio signal")
    parser.add_argument(
        "--hf_token", help="HuggingFace token for using pyannote Pipeline"
    )
    parser.add_argument("--device", help="Device: cpu / cuda")
    parser.add_argument("--fp16", help="Run in FP16 mode", default=True)
    parser.add_argument(
        "--batch_size", help="Batch size for acoustic model inference", default=10
    )
    return parser.parse_args()


def main(
    model_config: str,
    model_weights: str,
    tokenizer_path: str,
    device: str,
    audio_path: str,
    hf_token: str,
    fp16: bool,
    batch_size: int = 10,
    transcript_path: str = 'output.json'
):
    batch_size = int(batch_size)
    # Initialize model
    config = OmegaConf.load(model_config)
    with open_dict(config):
        config.tokenizer.dir = tokenizer_path

    model = EncDecRNNTBPEModel.from_config_dict(config)

    logging.basicConfig(level = logging.INFO)
    logging.getLogger("nemo_logger").setLevel(logging.FATAL)
    ckpt = torch.load(model_weights, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device)
    if device != "cpu" and fp16:
        model = model.float()
        model.preprocessor = model.preprocessor.float()
    model.eval()

    # Segment audio
    segments, boundaries = segment_audio(audio_path)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if torch.xpu.is_available():
        torch.xpu.empty_cache()
    # Transcribe segments
    transcriptions = []
    if device != "cpu" and fp16:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            transcriptions = model.transcribe(segments, batch_size=batch_size)[0]
    else:
        transcriptions = model.transcribe(segments, batch_size=batch_size)[0]

    # transcriptions = model.transcribe(segments, batch_size=batch_size)[0]   
    transcriptions = add_silence_line_breaks(transcriptions, boundaries)

    print("Transcriptions ended.")
    # print("Remove temporary files...")
    # for temp_file in segments:
    #         try:
    #             os.remove(temp_file)
    #         except OSError:
    #             pass

    # print("Removed temporary files.")
    # print("transcriptions", transcriptions)

    print("Unloading CTC model...")
    # del model.model
    # model.cpu()
    del model
    print("CTC model unloaded.")

    print("Sage model loading...")
    from sage.spelling_correction import AvailableCorrectors
    # from sage.spelling_correction import T5ModelForSpellingCorruption
    from transformers import T5ForConditionalGeneration, T5TokenizerFast

    # spelling_model = T5ModelForSpellingCorruption.from_pretrained(AvailableCorrectors.sage_mt5_large.value)
    spelling_model = T5ForConditionalGeneration.from_pretrained(AvailableCorrectors.sage_mt5_large.value)
    spelling_tokenizer = T5TokenizerFast.from_pretrained(AvailableCorrectors.sage_mt5_large.value)
    spelling_model.to(torch.device(device))
    # if device != "cpu" and fp16:
        # model = ipex.optimize(model)
        # spelling_model = ipex.optimize(spelling_model)
        # spelling_model = spelling_model.type(torch.float16)
        # spelling_model.to(torch.device(device))
        # spelling_model.half()
        # spelling_tokenizer = spelling_tokenizer.float()

    # spelling_model.eval()

    def batch_correct(
            sentences: List[str],
            batch_size: int,
            prefix: Optional[str] = "",
            **generation_params,
    ) -> List[List[Any]]:
        batches = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]
        result = []
        pb = tqdm(total=len(batches), desc="Batch correcting for transcriptions...")
        device = spelling_model.device
        for batch in batches:
            batch_prefix = [prefix + sentence for sentence in batch]
            with torch.inference_mode():
                encodings = spelling_tokenizer.batch_encode_plus(
                    batch_prefix, max_length=None, padding="longest", truncation=False, return_tensors='pt')
                for k, v in encodings.items():
                    encodings[k] = v.to(device)
                generated_tokens = spelling_model.generate(
                    **encodings, **generation_params, max_length=encodings['input_ids'].size(1) * 1.5)
                ans = spelling_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            result.append(ans)
            pb.update(1)
        return result

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

    # print("transcriptions", transcriptions)

    proceeded_transcriptions = process_transcriptions(transcriptions)

    flattened_list = flatten_list_of_lists(proceeded_transcriptions)

    # print("proceeded_transcriptions", proceeded_transcriptions)
    # print("flattened_list", flattened_list)

    # if device != "cpu" and fp16:
    #     with torch.autocast(device_type=device, dtype=torch.float16):
    #         corrected_strings = batch_correct(flattened_list, batch_size=batch_size)
    # else:
    #     corrected_strings = batch_correct(flattened_list, batch_size=batch_size)
    corrected_strings = batch_correct(flattened_list, batch_size=batch_size)

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
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        audio_path=args.audio_path,
        hf_token=args.hf_token,
        fp16=args.fp16,
        batch_size=args.batch_size,
    )