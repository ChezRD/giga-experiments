
import intel_extension_for_pytorch
import torch
import os
import logging
logging.basicConfig(level=logging.WARNING, force=True)
# torch.cuda.memory.change_current_allocator( torch.cuda.memory.CUDAPluggableAllocator('/usr/local/lib/alloc.so','gtt_alloc','gtt_free') )

import argparse
import torchaudio
import json
import yt_dlp
from tqdm import tqdm
from pathlib import Path


import numpy as np
from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessor as NeMoAudioToMelSpectrogramPreprocessor,
)
from nemo.collections.asr.parts.preprocessing.features import (
    FilterbankFeaturesTA as NeMoFilterbankFeaturesTA,
)

# from sbert_punc_case_ru import SbertPuncCase

from rupunkt import RUPunkt
from sage import Sage
from summarizer import Summarizer

import math
from pydub import AudioSegment
from pydub.effects import normalize

def split_wav(input_file, output_dir, chunk_size=35.0, padding=0.05, increase_db=2):
    """
    Splits a WAV file into smaller chunks with optional volume increase, padding, and normalization.

    Parameters:
    - input_file (str): Path to the input WAV file.
    - output_dir (str): Directory where the output chunks will be saved.
    - chunk_size (float): Size of each chunk in seconds. Default is 35 seconds.
    - padding (float): Padding to add at the end of each chunk in seconds. Default is 0.15 seconds.
    - increase_db (int): Amount to increase the volume in decibels. Default is 1 dB.

    Returns:
    - List[str]: List of file paths to the generated chunks.
    """
    try:
        # Load the audio file
        sound = AudioSegment.from_wav(input_file)
        
        # Increase the volume
        sound = sound + increase_db

        # Convert chunk size and padding from seconds to milliseconds
        chunk_size_ms = chunk_size * 1000
        padding_ms = padding * 1000
        
        # Calculate the number of chunks
        chunk_count = math.ceil(len(sound) / chunk_size_ms)

        # Normalize the volume
        normalized_sound = normalize(sound)

        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # List to store the names of the generated chunks
        chunk_names = []

        # Process each chunk with a progress bar
        for i in tqdm(range(chunk_count), desc="Splitting audio into smaller chunks"):
            start_time = i * chunk_size_ms
            end_time = min((i + 1) * chunk_size_ms + padding_ms, len(normalized_sound))
            chunk = normalized_sound[start_time:end_time]
            chunk = chunk.set_frame_rate(16000).set_channels(1)
            
            chunk_name = f"{output_dir}/chunk_{i}.wav"
            chunk.export(chunk_name, format="wav")
            chunk_names.append(chunk_name)

        return chunk_names

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

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

def download_youtube_to_wav(url, output_dir="downloads", filename="test", sample_rate=16000):
    # Create the output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set the output filename
    if filename is None:
        filename = "%(title)s.%(ext)s"
    else:
        filename = f"{filename}.%(ext)s"

    # Download the video using yt_dlp
    ydl_opts = {
        'outtmpl': str(output_dir / filename),
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as e:
        logging.error(f"Error downloading video: {e}")
        return None

    output_path = os.path.abspath(output_dir / filename.replace('%(ext)s', 'wav'))
    logging.info(f"Downloaded {output_path} as WAV")
    return str(output_path)

def chunks(iterable, duration, sample_rate):
    total_samples = len(iterable)
    chunk_size = int(sample_rate * duration)
    num_chunks = total_samples // chunk_size
    remainder = total_samples % chunk_size

    for i in range(num_chunks):
        yield iterable[i * chunk_size: (i + 1) * chunk_size]
    if remainder > 0:
        yield iterable[-remainder:]

def offset_to_time(seconds):
    # Convert offset to seconds
    # seconds = offset / 10

    # Calculate hours, minutes, and seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

class NpJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpJsonEncoder, self).default(obj)


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
        self.featurizer = FilterbankFeaturesTA(  # Deprecated arguments; kept for config compatibility
            mel_scale=mel_scale, **kwargs,
        )

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference using GigaAM-CTC checkpoint"
    )
    parser.add_argument("--model_config", help="Path to GigaAM-CTC config file (.yaml)")
    parser.add_argument(
        "--model_weights", help="Path to GigaAM-CTC checkpoint file (.ckpt)"
    )
    parser.add_argument("--youtube_url", help="Path to audio signal")
    parser.add_argument("--device", help="Device: cpu / cuda")
    parser.add_argument("--chunk_duration", type=float, default=29.57, help="Duration of each audio chunk in seconds")

    return parser.parse_args()

def main(model_config: str, model_weights: str, device: str, youtube_url: str, chunk_duration: float):

    file_name = download_youtube_to_wav(youtube_url, "downloads", "nts.wav")

    chunk_files = split_wav(file_name, "temp", chunk_duration)

    # audio_data, sample_rate = torchaudio.load(file_name, backend="ffmpeg")

    # if sample_rate != 16000:
    #     audio_data = torchaudio.functional.resample(audio_data, orig_freq=sample_rate, new_freq=16000)
    #     sample_rate = 16000

    model = EncDecCTCModel.from_config_file(model_config)

    ckpt = torch.load(model_weights, map_location=torch.device('xpu'))
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device)
    model.eval()

    # transcriptions = []
    # temp_audio_paths = []
    
    # for idx, chunk in enumerate(chunks(audio_data[0], chunk_duration, sample_rate)):
    #     audios = torch.unsqueeze(chunk, 0)
    #     temp_audio_path = f"./temp/temp_chunk_{idx}.wav"
    #     temp_audio_paths.append(temp_audio_path)
    #     torchaudio.save(temp_audio_path, audios, sample_rate)
    
    # del audio_data

    # Transcribe each chunk
    hypothesis = model.transcribe(chunk_files, batch_size=16, return_hypotheses=True)

    punktuation_model = RUPunkt()
    # punktuation_model = punktuation_model.model.to(device)

    spelling_model = Sage(device)

    transcriptions = []
    timesteps = []
    words = []
    segments = []

    chunk_id = 0
    
    for entry in tqdm(hypothesis, desc="Processing transcribed segments"):
        
        timesteps.append(entry.timestep['timestep'])
        words.append(entry.timestep['word'])

        word_list = entry.timestep['word']
        if word_list:
            word_tokens = []

            if word_list[0]['start_offset'] >= 25:
                word_tokens.append("...")

            for i in range(len(word_list)):
                current_word = word_list[i]
                word_tokens.append(current_word['word'])
                
                # Check the offset difference if it's not the first word
                if i > 0:
                    previous_word = word_list[i - 1]
                    if current_word['start_offset'] - previous_word['end_offset'] >= 25:
                        word_tokens.append("...")

            first_word = entry.timestep['word'][0]
            last_word = entry.timestep['word'][-1]

            last_timestep = entry.timestep['timestep'][-1]

            step_to_seconds = chunk_duration / last_timestep

            start_offset = first_word['start_offset'] * step_to_seconds + chunk_id * chunk_duration
            end_offset = last_word['end_offset'] * step_to_seconds + chunk_id * chunk_duration

            start_time = offset_to_time(start_offset)
            end_time = offset_to_time(end_offset)

            segments.append({
                "start": start_time,
                "end": end_time,
                "text": punktuation_model.punctuate(" ".join(word_tokens)),
                "raw": " ".join(word_tokens)
                # "text": entry.text
            })
            transcriptions.append(" ".join(word_tokens))

        chunk_id += 1

    for temp_audio in tqdm(chunk_files, desc="Removing temporary audio files"):
        os.remove(temp_audio)  # Remove temporary audio file

    transcript_path = "output.json"

    print(f"Doing some AI magic on parsed texts...")
    # del punktuation_model.model
    del punktuation_model

    resulting_text = ' '.join(spelling_model.fix_spelling(transcriptions))
    
    del spelling_model.model
    del spelling_model


    # summarizer_model = Summarizer(device)

    with open(transcript_path, "w", encoding="utf8") as fp:
        result = {
            "segments": segments,
            "words": words,
            "timesteps": timesteps,
            # "text": ' '.join(transcriptions),
            "text": resulting_text,
            # "summary": summarizer_model.summarize(resulting_text, n_words=2000)
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
        youtube_url=args.youtube_url,
        chunk_duration=args.chunk_duration,
    )
