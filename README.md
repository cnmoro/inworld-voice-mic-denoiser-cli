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
| NVIDIA RTX 5060 Ti (Windows) | `--gpu` (Vulkan) | 11.2x |
| NVIDIA RTX 2060 SUPER (Linux) | `--gpu` (Vulkan) | 5.6x |
| AMD Ryzen 9 9950X3D, 16 cores, AVX-512 (Windows) | `--cpu` | 4.7x |
| AMD Ryzen 7 3700X, 8 cores, AVX2 (Linux) | `--cpu` | 1.0x |
| Apple M4, MacBook Air in Low Power Mode (macOS) | `--gpu` (Metal) | 0.9x |
| Apple M4, MacBook Air in Low Power Mode (macOS) | `--cpu` | 0.8x |

These are for a 62 s mono file. Stereo is denoised channel by channel, so it takes twice as long unless you pass
`--downmix`.

This is a 64M-parameter model that needs about 3.7 TFLOP of matrix products per 8 s window, so the speed is set by
how many multiply-adds per second the hardware can do, not by memory: see [Performance notes](#performance-notes).

## Building

```
cargo build --release          # needs Rust 1.89+, and a C compiler for mimalloc
```

The GPU backend is always included (wgpu); there are no feature flags.

## How it works

The network (STFT, 46 band tokens, 4 attention and 3 bidirectional Mamba blocks, per-band mask/residual heads, iSTFT)
was re-implemented in Rust from the reference code:

* `src/cpu.rs`: CPU backend. Consecutive row-local layers run as one chain over a few dozen tokens at a time, attention
  is computed per sequence and head in cache-sized slabs, and each Mamba direction streams a sequence through
  projection, convolution, scan and output projection chunk by chunk, so intermediates stay in L2 and only the token
  matrix goes through main memory.
* `src/gemm.rs`: the matrix product: weights packed once at load time, AVX-512 (12x32), AVX2+FMA (6x16) and NEON (8x12)
  microkernels picked at run time. On Apple silicon the products go to Accelerate, which drives the matrix
  coprocessor (AMX/SME) that NEON code cannot reach.
* `src/simd.rs`: the other CPU kernels (layer norm, softmax, GELU, SiLU, softplus, causal conv, selective scan), written
  once and compiled for the baseline, AVX2 and AVX-512.
* `src/gpu.rs` + `src/kernels.rs`: GPU backend on wgpu with hand-written WGSL kernels (tiled matmul, layer norm,
  softmax, rotary attention, causal conv, selective scan). The matmul kernel is generated for a tile shape chosen per
  GPU vendor and per matrix shape. All weights stay resident in VRAM.
* `src/pipeline.rs`: the reference's 8 s windows with 2 s linear crossfades, streamed from disk so memory stays bounded.
* `src/media.rs`: ffmpeg/ffprobe glue.

Both backends reproduce the upstream golden fixtures (`tests/golden.py`) with a maximum error of about 1.5e-5
(the limit is 1e-4; the residual is 16-bit output rounding), and agree with the PyTorch reference to about 1e-6 on
longer, chunked clips.

```
cargo test                                             # unit tests
python3 tests/golden.py target/release/inworldvoice --cpu   # needs ffmpeg + numpy
```

## Performance notes

A window is 36846 tokens of width 512: about 3.7 TFLOP of matrix products. On the CPU path the data that has to cross
main memory is roughly 9 GB per window (estimated from the buffer sizes), which at the speeds above is 1 to 7 GB/s on
machines whose memory delivers 17 to 100 GB/s: the CPU backend is limited by arithmetic, not by memory bandwidth.
The GPU matmul kernel reads about 115 GB from its buffers per window, most of it served by the GPU's caches; that is
100 to 200 GB/s on cards with 448 GB/s of memory bandwidth. What is reached:

| Hardware | Matrix products in the model | For comparison |
|---|---|---|
| RTX 5060 Ti | 9.7 TFLOP/s in the large products | 23.7 TFLOP/s nominal fp32 peak |
| RTX 2060 SUPER | 4.7 TFLOP/s in the large products | 7.2 TFLOP/s nominal fp32 peak |
| Ryzen 9 9950X3D | 3.5 TFLOP/s in the feed-forward chains, 2.9 TFLOP/s over the whole window | 3.9 TFLOP/s for the same kernel on data that never leaves L2 |
| Ryzen 7 3700X | 0.70 TFLOP/s in the chains, 0.57 TFLOP/s over the window | 0.78 TFLOP/s in L2 |
| Apple M4 GPU (Low Power Mode) | 0.83 TFLOP/s | 0.75 TFLOP/s for PyTorch MPS `matmul` on the same machine |
| Apple M4 CPU (Low Power Mode) | 0.8 to 1.0 TFLOP/s in the chains | 1.1 to 1.4 TFLOP/s for Accelerate `sgemm` alone |

The GPU kernels use fp32 throughout. wgpu does not expose tensor cores (cooperative matrices), and half precision
would not meet the 1e-4 accuracy limit, so the remaining distance to the nominal GPU peak is what a portable shader
leaves on the table.

For measuring and tuning:

* `INWORLD_PROFILE=1` prints the time per kernel (GPU) or per stage (CPU) for every window, with the GFLOP/s of the
  matrix products.
* `inworldvoice --bench` prints the matrix-product rate of one core and of all cores, the scan rate and the memory
  bandwidth of the machine.
* `INWORLD_MM="kind,tx,ty,rx,ry,ks[;...]"` overrides the GPU matmul tile shape (`kind` is `s`, `v` or `d`, see
  `src/kernels.rs`); `INWORLD_GEMM="kc,mc,blk"` the CPU cache blocking; `INWORLD_ISA=avx2|neon|generic` forces a
  narrower CPU kernel.

## What the model does

It removes noise, music and other background sound and keeps the foreground voice, including breaths and laughter.
It does not restore missing bandwidth, and when two people talk at once the louder one is kept. See the
[model card](https://huggingface.co/inworld/voice-isolator).

## License

Apache-2.0. The model and the algorithm are by Inworld AI (also Apache-2.0); see `NOTICE`.
