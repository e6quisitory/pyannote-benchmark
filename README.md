# pyannote.audio NVIDIA Benchmark
Uses some sample videos from YouTube of varying duration to benchmark [pyannote.audio speaker diarization](https://huggingface.co/pyannote/speaker-diarization-3.1) on an NVIDIA GPU.

More YouTube video id's can be added in `video_ids.txt`.

### Running Benchmark
Have `pyannote.audio` installed, of course, as well as `yt-dlp` and `ffmpeg`.  

Also obtain a Hugging Face token for `pyannote.audio` speaker diarization and put it in the script (line `103`).

Then simply run the script, passing in the GPU number you'd like to benchmark (obtain from `nvidia-smi`).
```
python3 benchmark.py <GPU_NUMBER>
```
### Past Benchmarks
I've run this script on some GPUs. If you've run it on a GPU not in this list, feel free to submit a PR and we can get it added.

Note: the benchmark I ran for each of these GPUs was with many more videos than the ones in `video_ids.txt`. I included videos ranging from a few minutes long to over 5 hours long. I have reduced the number of videos in `video_ids.txt` for this repo so that people can get a quick benchmark result.

| GPU      | Diarization Processing Secs / 1 Audio Sec |
| ----------- | ----------- |
| H100 PCIe      | 0.01150315       |
| 4090      | 0.014636619       |
| A100   | 0.015494026       |
| 3090   | 0.017570489       |
| A10G   | 0.023336284       |
| 1080 Ti   | 0.028720307      |
| 3060 Ti   | 0.029580563     |
| 980 Ti   | 0.038951152    |
| RTX A4000   | 0.039963005    |
| 1070 Ti   | 0.040482773   |
| Apple M1   | 0.147083095  |

_For Apple M1_ --> `pipeline.to(torch.device("mps"))` (I used a seperate script)