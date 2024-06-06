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

def generate_timestamps(start, end, sampling_rate=16000, max_duration=23.0, min_duration=12.0, new_chunk_threshold=0.18):
    timestamps = []
    total_duration = (end - start) / sampling_rate  # Total duration in seconds
    num_segments = int(total_duration / max_duration) + 1  # Number of segments required

    if total_duration < max_duration:
        timestamps.append({"start": max(int(start - (new_chunk_threshold * sampling_rate)), 0), "end": int(end + (new_chunk_threshold * sampling_rate))})
    else:
        segment_duration = total_duration / num_segments
        segment_start = max(int(start - (new_chunk_threshold * sampling_rate)), 0)
        for _ in range(num_segments):
            segment_end = int(segment_start + (segment_duration * sampling_rate) + (new_chunk_threshold * sampling_rate))
            timestamps.append({"start": segment_start, "end": segment_end})
            segment_start = int(segment_end - (new_chunk_threshold * sampling_rate))

    return timestamps

def segment_audio(
    transcription_model: EncDecRNNTBPEModel,
    audio_path: str,
    device = "cpu",
    fp16 = False,
    batch_size = 10,
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

    vad_iterator = VADIterator(model)

    window_size_samples = 1024 # number of samples in a single audio chunk

    resulting_list = []

    from contextlib import suppress

    current_segment_start = 0

    for i in tqdm(range(0, len(audio), window_size_samples), desc="Processing audio..." ):
        with suppress(Exception):
            speech_dict = vad_iterator(audio[i: i+ window_size_samples], return_seconds=False)

            if speech_dict:
                if 'start' in speech_dict:
                    current_segment_start = speech_dict['start']
                    # Perform action for start of speech fragment
                    # print(f"Start of speech at {current_segment_start} samples")
                elif 'end' in speech_dict:
                    # chunks = []
                    current_segment_end = speech_dict['end']
                    # Perform action for end of speech fragment
                    # print(f"End of speech at {current_segment_end} samples")
                    
                    timestamps = generate_timestamps(current_segment_start, current_segment_end, sampling_rate)

                    # print("timestamps", timestamps)

                    # for time_part in timestamps:
                    #     chunks.append(audio[time_part["start"]:time_part["end"]])

                    chunks = collect_chunks(timestamps, audio)
                    chunks.to(device)

                    if device != "cpu" and fp16:
                        with torch.autocast(device_type="cuda", dtype=torch.float16), torch.no_grad():
                            transcripts = transcription_model.transcribe(chunks, batch_size=batch_size, verbose=False)[0]
                    else:
                        with torch.no_grad():
                            transcripts = transcription_model.transcribe(chunks, batch_size=batch_size, verbose=False)[0]

                    resulting_list.append({
                        "start": current_segment_start / sampling_rate,
                        "end": current_segment_end / sampling_rate,
                        "transcripts": transcripts
                    })

                    del chunks
                    torch.cuda.empty_cache()

                    current_segment_start = current_segment_end

    vad_iterator.reset_states() # reset model states after each audio

    del vad_iterator, audio, get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks

    return resulting_list


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
    # with torch.autocast(device_type="cuda", dtype=torch.float16):
    resulting_list = segment_audio(model, audio_path, device, fp16, batch_size)

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if torch.xpu.is_available():
        torch.xpu.empty_cache()
    
    # print("resulting_list", resulting_list)

    # Calculate the pause durations
    pauses = []

    for i in range(1, len(resulting_list)):
        pause = resulting_list[i]['start'] - resulting_list[i-1]['end']
        resulting_list[i]['pause'] = pause
        pauses.append(pause)

    mean_pause = sum(pauses) / len(pauses)

    for i in range(1, len(resulting_list)):
        if resulting_list[i]['pause'] >= min(mean_pause, 0.38):
            resulting_list[i-1]['transcripts'][-1] += "."

    export_segments = []

    for item in resulting_list:
        export_segments.append({
            "start": format_time(item["start"]),
            "end": format_time(item["end"]),
            "pause": format_time(item.get('pause', 0)),
            "text": ' '.join(item['transcripts']).strip()
        })
    # print("pauses", pauses)

    # print("resulting_list", resulting_list)

    transcriptions = " ".join([' '.join(item['transcripts']).strip() for item in resulting_list]).strip()

    # print("transcriptions", transcriptions)

    from rupunkt import RUPunkt

    punktuation = RUPunkt(device)

    def capitalize_first_letter(string):
        return string[0].upper() + string[1:]

    def recursive_split(string, min_length=128, max_length=256):
        """
        Recursively split a string into parts with lengths between min_length and max_length.
        Primary split by '.', secondary split by the nearest space around the middle.
        """
        # # If the string is shorter than min_length, return it as is
        # if len(string) < min_length:
        #     return [string]

        # If the string is within the acceptable length, return it as is
        if len(string) <= max_length:
            return [punktuation.punctuate(capitalize_first_letter(string))]

        # Try to split the string by the nearest '.' within max_length range
        split_index = string.rfind('.', 0, max_length)
        if split_index != -1:
            first_part = string[:split_index + 1].strip()
            second_part = string[split_index + 1:].strip()
            return recursive_split(first_part, min_length, max_length) + recursive_split(second_part, min_length, max_length)

        # If no '.' found, split by the nearest space around the middle
        split_index = len(string) // 2
        for i in range(split_index, len(string)):
            if string[i] == ' ':
                split_index = i
                break
        for i in range(split_index, 0, -1):
            if string[i] == ' ':
                if split_index < i:
                    split_index = i
                    break

        first_part = string[:split_index].strip()
        second_part = string[split_index:].strip()
        
        return recursive_split(capitalize_first_letter(first_part), min_length, max_length) + recursive_split(capitalize_first_letter(second_part), min_length, max_length)
    
    # result = split_into_chunks(transcriptions, 256)
    result = recursive_split(transcriptions, 128, 368)
    # result = concatenate_and_split(transcriptions, 500)

    # result_str = '\n'.join([' '.join(sublist) for sublist in result])

    # print("result=", result)

    # print("Sage model loading...")

    from sage.spelling_correction import AvailableCorrectors
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    spelling_model = T5ForConditionalGeneration.from_pretrained(AvailableCorrectors.sage_fredt5_large.value)
    spelling_tokenizer = AutoTokenizer.from_pretrained(AvailableCorrectors.sage_fredt5_large.value)
    spelling_model.to(torch.device(device))
    spelling_model.eval()


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

    # print("Sage model loaded.")

    corrected_strings = batch_correct(result, batch_size=batch_size)

    # print("corrected_strings", corrected_strings)
    flatten_result = []

    for item in corrected_strings:
        flatten_result.append("\n".join(item))

    with open(transcript_path, "w", encoding="utf8") as fp:
        result = {
            "segments": export_segments,
            # "result": result,
            # "flatten_result": flatten_result,
            # "corrected_strings": corrected_strings,
            "text": "\n".join(flatten_result),
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