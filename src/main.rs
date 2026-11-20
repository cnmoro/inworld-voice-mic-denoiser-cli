#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod config;
mod dsp;
mod fast;
mod gpu;
mod kernels;
mod media;
mod model;
mod pipeline;
mod weights;

use anyhow::{bail, Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Remove noise from the voice in any audio or video file.
///
/// Writes <name>_denoised.<ext> next to each input, keeping the original container,
/// video, subtitles and audio codec settings. Needs ffmpeg and ffprobe on PATH.
#[derive(Parser)]
#[command(version, about)]
struct Args {
    /// Input files (mp3, wav, ogg, flac, m4a, mp4, mkv, mov, webm, ... anything ffmpeg reads)
    #[arg(required = true)]
    files: Vec<PathBuf>,
    /// Run on the CPU
    #[arg(long, conflicts_with = "gpu")]
    cpu: bool,
    /// Run on the GPU (Vulkan, DirectX 12 or Metal; no CUDA needed). Default: GPU if available, else CPU
    #[arg(long)]
    gpu: bool,
    /// GPU index for --gpu
    #[arg(long, default_value_t = 0)]
    gpu_index: usize,
    /// Output path (only with a single input file)
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Use a local model.safetensors instead of the downloaded one
    #[arg(long)]
    model: Option<PathBuf>,
    /// Path to ffmpeg (binary or its directory)
    #[arg(long)]
    ffmpeg: Option<String>,
    /// CPU threads (default: all cores)
    #[arg(long)]
    threads: Option<usize>,
    /// Denoise a mono mix and write it to every channel (faster for multi-channel audio)
    #[arg(long)]
    downmix: bool,
    /// Directory for temporary raw audio (default: system temp). Needs ~4 bytes x samples.
    #[arg(long)]
    tmp_dir: Option<PathBuf>,
}

/// Loads the model on the requested backend: --cpu, --gpu, or auto (GPU if usable, else CPU).
fn load_backend(args: &Args, weights: &Path) -> Result<(Box<dyn config::Denoiser>, String)> {
    let cpu = || -> Result<(Box<dyn config::Denoiser>, String)> {
        Ok((Box::new(model::Model::load(weights)?), "CPU".to_string()))
    };
    if args.cpu {
        return cpu();
    }
    match gpu::GpuModel::new(weights, args.gpu_index) {
        Ok((m, name)) => Ok((Box::new(m), format!("GPU: {name}"))),
        Err(e) if args.gpu => Err(e.context("--gpu requested but the GPU backend could not start")),
        Err(e) => {
            eprintln!("GPU unavailable ({e:#}); using the CPU");
            cpu()
        }
    }
}

fn output_path(input: &Path) -> Result<PathBuf> {
    let stem = input.file_stem().context("input has no file name")?.to_string_lossy();
    let name = match input.extension() {
        Some(ext) => format!("{stem}_denoised.{}", ext.to_string_lossy()),
        None => format!("{stem}_denoised"),
    };
    Ok(input.with_file_name(name))
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.output.is_some() && args.files.len() > 1 {
        bail!("--output works with a single input file");
    }
    if let Some(t) = args.threads {
        rayon::ThreadPoolBuilder::new().num_threads(t).build_global()?;
    }
    let ff = media::Ffmpeg::locate(args.ffmpeg.as_deref())?;
    // Fail on bad inputs before loading the model.
    for f in &args.files {
        if !f.is_file() {
            bail!("{} is not a file", f.display());
        }
    }
    let weights = weights::resolve(args.model.as_deref())?;
    eprintln!("Loading model...");
    let (model, dev_name) = load_backend(&args, &weights)?;
    eprintln!("Running on {dev_name}");
    let rate = model.cfg().sample_rate as u32;

    for input in &args.files {
        let dest = match &args.output {
            Some(o) => o.clone(),
            None => output_path(input)?,
        };
        if dest == *input {
            bail!("refusing to overwrite the input file");
        }
        let t0 = Instant::now();
        let streams = ff.probe(input)?;
        if streams.is_empty() {
            bail!("{} has no audio stream", input.display());
        }
        let tmp = match &args.tmp_dir {
            Some(d) => tempfile::tempdir_in(d)?,
            None => tempfile::tempdir()?,
        };
        let mut done = Vec::new();
        let mut seconds = 0.0;
        for s in &streams {
            let raw_in = tmp.path().join(format!("in{}.raw", s.ordinal));
            let raw_out = tmp.path().join(format!("out{}.raw", s.ordinal));
            ff.decode(input, s, &raw_in, rate)?;
            let n = std::fs::metadata(&raw_in)?.len() as usize / 4 / s.channels;
            let passes = if args.downmix { 1 } else { s.channels };
            let bar = ProgressBar::new((n * passes) as u64);
            bar.set_style(ProgressStyle::with_template("{prefix} [{bar:40}] {percent}% eta {eta}")?);
            bar.set_prefix(format!("{} audio {}", input.file_name().unwrap().to_string_lossy(), s.ordinal));
            let n = pipeline::denoise_raw(model.as_ref(), &raw_in, s.channels, &raw_out, args.downmix, |d| bar.set_position(d as u64))?;
            bar.finish_and_clear();
            let _ = std::fs::remove_file(&raw_in);
            seconds += n as f64 / rate as f64;
            done.push((s.clone(), raw_out));
        }
        ff.mux(input, &done, &dest, rate)?;
        let el = t0.elapsed().as_secs_f64();
        eprintln!("{} -> {}  ({:.1}s of audio in {:.1}s, {:.2}x real time on {})",
            input.display(), dest.display(), seconds, el, seconds / el, dev_name);
    }
    Ok(())
}
