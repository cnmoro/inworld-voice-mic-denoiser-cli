# inworldvoice

Remove background noise from the voice in any audio or video file, from the command line.

```
inworldvoice interview.mp4        # writes interview_denoised.mp4
inworldvoice --gpu podcast.mp3    # writes podcast_denoised.mp3
inworldvoice --cpu voice.wav      # writes voice_denoised.wav
```

It runs the [Inworld Voice Isolator](https://huggingface.co/inworld/voice-isolator) v1 offline model
(64M parameters, 48 kHz, Apache-2.0) as a standalone Rust program: no Python, no PyTorch, no CUDA.
The output is always the original file with its audio replaced by the cleaned version, in the **same
container and extension**. Video, subtitles, chapters, metadata, extra audio tracks and the original audio codec,
bitrate, sample rate and channel layout are kept.

It is meant as a post-processing step. The model is bidirectional and works on 8 s windows, so it is not a
real-time filter.

## Install

```
pip install inworldvoice        # prebuilt wheels for Linux, Windows and macOS (Intel and Apple silicon)
```

This puts the `inworldvoice` command on your PATH (the wheel contains the native binary; there is no Python code).

## Requirements

* **ffmpeg and ffprobe** on `PATH` (or pass `--ffmpeg <path>`). They do all decoding and muxing, so any format ffmpeg
  reads works: mp3, wav, ogg, opus, flac, m4a, mp4, mkv, mov, webm, avi, ...
* For `--gpu`: any GPU with a Vulkan, DirectX 12 or Metal driver (NVIDIA, AMD, Intel, Apple). Nothing else to install.
  About 2 GB of free VRAM is needed.
* The first run downloads the model weights (about 256 MB) to your cache directory and checks their SHA-256.
* Temporary disk space of roughly 4 bytes x 48000 x seconds x channels (about 700 MB per hour of stereo audio).

## Usage

```
inworldvoice [OPTIONS] <FILES>...

  --cpu               Run on the CPU
  --gpu               Run on the GPU
  --gpu-index <N>     Which GPU to use with --gpu (default 0, discrete GPUs first)
  -o, --output <PATH> Output path (single input only)
  --downmix           Denoise a mono mix and write it to every channel (faster for multi-channel audio)
  --model <PATH>      Use a local denoiser.safetensors instead of the downloaded one
  --ffmpeg <PATH>     ffmpeg binary or its directory
  --threads <N>       CPU threads (default: all cores)
  --tmp-dir <DIR>     Where to put temporary raw audio
```

With neither `--cpu` nor `--gpu` it uses the GPU if one works and falls back to the CPU. Several files can be given at once
and the model is loaded once. An existing `*_denoised.*` file is overwritten.

Every audio track of a file is processed. Multi-channel audio is denoised one channel at a time (use `--downmix` to
process a mono mix instead, which is faster but makes the channels identical).

## Speed

Real-time factor = audio duration / processing time (higher is faster):

| Hardware | Backend | Speed |
|---|---|---|
| NVIDIA RTX 5060 Ti (Windows) | `--gpu` (Vulkan) | 6.0x |
| NVIDIA RTX 2060 SUPER (Linux) | `--gpu` (Vulkan) | 3.2x |
| AMD Ryzen 9 9950X3D, 16 cores (Windows) | `--cpu` | 0.9x |
| AMD Ryzen 7 3700X, 8 cores (Linux) | `--cpu` | 0.4x |

These are for mono audio. Stereo is denoised channel by channel, so it takes twice as long unless you pass `--downmix`.

The CPU path is 20x faster than the reference PyTorch code on the same machine, but this is a 64M-parameter model
(about 4 TFLOP per 8 s of audio), so a GPU is strongly recommended for anything long.

## Building

```
cargo build --release          # needs Rust 1.80+, and a C compiler for mimalloc
```

The GPU backend is always included (wgpu); there are no feature flags.

## How it works

The network (STFT, 46 band tokens, 4 attention and 3 bidirectional Mamba blocks, per-band mask/residual heads, iSTFT)
was re-implemented in Rust from the reference code:

* `src/model.rs`: CPU backend. Matrix products use candle's `gemm`; the rest are fused, multithreaded f32 kernels
  (`src/fast.rs`) including a vectorised selective scan.
* `src/gpu.rs` + `src/kernels.rs`: GPU backend on wgpu with hand-written WGSL kernels (tiled matmul, layer norm,
  softmax, rotary attention, causal conv, selective scan). All weights stay resident in VRAM.
* `src/pipeline.rs`: the reference's 8 s windows with 2 s linear crossfades, streamed from disk so memory stays bounded.
* `src/media.rs`: ffmpeg/ffprobe glue.

Both backends reproduce the upstream golden fixtures (`tests/golden.py`) with a maximum error of about 1.5e-5
(the limit is 1e-4; the residual is 16-bit output rounding), and agree with the PyTorch reference to about 1e-6 on
longer, chunked clips.

```
cargo test                                             # unit tests
python3 tests/golden.py target/release/inworldvoice --cpu   # needs ffmpeg + numpy
```

## What the model does

It removes noise, music and other background sound and keeps the foreground voice, including breaths and laughter.
It does not restore missing bandwidth, and when two people talk at once the louder one is kept. See the
[model card](https://huggingface.co/inworld/voice-isolator).

## License

Apache-2.0. The model and the algorithm are by Inworld AI (also Apache-2.0); see `NOTICE`.
