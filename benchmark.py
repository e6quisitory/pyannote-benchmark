import subprocess
import shutil
import os
import sys
import time
import shutil
import csv
import torch
import torchaudio
from tqdm import tqdm
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from sklearn.linear_model import LinearRegression
import numpy as np

# Get video duration using yt-dlp
def get_video_duration(video_id):
    command = [
        'yt-dlp',
        '--get-duration',
        '--no-warnings',
        f'https://www.youtube.com/watch?v={video_id}'
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    duration_str = result.stdout.decode('utf-8').strip()
    h, m, s = 0, 0, 0
    if ':' in duration_str:
        parts = duration_str.split(':')
        if len(parts) == 3:
            h, m, s = map(int, parts)
        elif len(parts) == 2:
            m, s = map(int, parts)
        else:
            s = int(parts[0])
    else:
        s = int(duration_str)
    return h * 3600 + m * 60 + s

########### j0: Download audio from YouTube ######################

def download_audio(video_id, download_dir):
    command = [
        'yt-dlp',
        '-f', '140/bestaudio',
        '--extract-audio',
        '--audio-format', 'm4a',
        '--audio-quality', '0',
        '-o', os.path.join(download_dir, '%(id)s.%(ext)s'),
        f'https://www.youtube.com/watch?v={video_id}',
        '--quiet'
    ]
    subprocess.run(command, check=True)

def convert_to_wav(input_file, output_file):
    command = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel',
        'error',
        '-i', input_file,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ac', '2',
        '-ar', '44100',
        output_file,
        '-y'
    ]
    subprocess.run(command, check=True)

def j0(video_id):
    start_time = time.time()
    working_dir = os.path.join(os.getcwd(), video_id)
    
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    
    os.makedirs(working_dir, exist_ok=True)
    wav_file = os.path.join(working_dir, 'audio.wav')

    print(f"{video_id} --> yt-dlp")
    download_audio(video_id, working_dir)
    
    input_file = os.path.join(working_dir, f"{video_id}.m4a")
    print(f"{video_id} --> ffmpeg")
    convert_to_wav(input_file, wav_file)
    
    os.remove(input_file)
    print(f"{video_id} --> j0 done")
    
    time_taken = time.time() - start_time
    return time_taken

############ j1: Run pyannote Diarization ########################

def j1(video_id, gpu_number):
    working_dir = os.path.join(os.getcwd(), video_id)
    wav_file = os.path.join(working_dir, 'audio.wav')

    start_time = time.time()

    device = torch.device(f"cuda:{gpu_number}" if torch.cuda.is_available() else "cpu")
    
    hugging_face_token = '' # Replace with your own
    
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hugging_face_token)
    pipeline.to(device)

    waveform, sample_rate = torchaudio.load(wav_file)
    with ProgressHook() as hook:
        diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, hook=hook)

    segments = []
    for segment, _, speaker in tqdm(diarization.itertracks(yield_label=True), desc="Processing segments", total=len(diarization)):
        segments.append({
            "speaker": speaker,
            "start": segment.start,
            "end": segment.end
        })

    shutil.rmtree(working_dir)

    time_taken = time.time() - start_time
    print(f"{video_id} --> j1 done")
    return time_taken

################# main ##########################

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python benchmark.py <GPU Number>")
        sys.exit(1)
    
    gpu_number = sys.argv[1]

    video_durations = []
    j1_times = []

    with open('benchmark_results.csv', mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Video ID', 'Video Duration (s)', 'J0 Time Taken', 'J1 Time Taken'])
        
        with open('video_ids.txt', 'r') as file:
            for line in file:
                video_id = line.strip()
                video_duration = get_video_duration(video_id)
                video_durations.append(video_duration)
                j0_time_taken = j0(video_id)
                j1_time_taken = j1(video_id, gpu_number)
                j1_times.append(j1_time_taken)
                writer.writerow([video_id, video_duration, round(j0_time_taken, 3), round(j1_time_taken, 3)])

    # Convert lists to numpy arrays for regression analysis
    video_durations_np = np.array(video_durations).reshape(-1, 1)
    j1_times_np = np.array(j1_times).reshape(-1, 1)

    # Perform linear regression
    model = LinearRegression().fit(video_durations_np, j1_times_np)
    processing_secs_per_video_sec = model.coef_[0][0]

    # Append regression result to CSV
    with open('benchmark_results.csv', mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Processing Secs / 1 Audio Sec', processing_secs_per_video_sec])

    print(f"Benchmark results and regression analysis have been written to benchmark_results.csv.")
    print(f"Processing Secs / 1 Audio Sec: {processing_secs_per_video_sec}")