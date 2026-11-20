//! Chunked offline rendering over raw f32 audio files, with bounded memory.

use crate::config::Denoiser;
use anyhow::Result;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

fn linspace(from: f32, to: f32, n: usize) -> Vec<f32> {
    if n == 1 {
        return vec![from];
    }
    (0..n).map(|i| from + (to - from) * i as f32 / (n - 1) as f32).collect()
}

/// Reads `len` frames starting at `start` and returns channel `ch` (or the channel mean).
fn read_channel(f: &mut File, channels: usize, start: usize, len: usize, ch: Option<usize>) -> Result<Vec<f32>> {
    f.seek(SeekFrom::Start((start * channels * 4) as u64))?;
    let mut bytes = vec![0u8; len * channels * 4];
    f.read_exact(&mut bytes)?;
    let mut out = Vec::with_capacity(len);
    for frame in bytes.chunks_exact(channels * 4) {
        let sample = |c: usize| f32::from_le_bytes(frame[c * 4..c * 4 + 4].try_into().unwrap());
        out.push(match ch {
            Some(c) => sample(c),
            None => (0..channels).map(sample).sum::<f32>() / channels as f32,
        });
    }
    Ok(out)
}

fn write_f32(w: &mut impl Write, xs: &[f32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(xs.len() * 4);
    for x in xs {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    Ok(w.write_all(&bytes)?)
}

/// Overlapping chunks with linear crossfades, exactly as the reference `enhance`.
fn enhance_stream(
    model: &dyn Denoiser,
    n: usize,
    mut read: impl FnMut(usize, usize) -> Result<Vec<f32>>,
    mut write: impl FnMut(&[f32]) -> Result<()>,
    mut tick: impl FnMut(usize),
) -> Result<()> {
    let chunk = model.cfg().chunk_samples();
    let overlap = model.cfg().overlap_samples();
    if n <= chunk {
        let y = model.run(&read(0, n)?)?;
        write(&y)?;
        tick(n);
        return Ok(());
    }
    let mut starts: Vec<usize> = (0..n - chunk).step_by(chunk - overlap).collect();
    starts.push(n - chunk);
    let (mut base, mut acc, mut wsum) = (0usize, Vec::<f32>::new(), Vec::<f32>::new());
    for (i, &s) in starts.iter().enumerate() {
        let e = s + chunk;
        let y = model.run(&read(s, chunk)?)?;
        let mut w = vec![1.0f32; chunk];
        if i > 0 {
            let fade_in = starts[i - 1] + chunk - overlap - s;
            w[..fade_in].iter_mut().for_each(|v| *v = 0.0);
            if overlap > 0 {
                w[fade_in..fade_in + overlap].copy_from_slice(&linspace(0.0, 1.0, overlap));
            }
        }
        if overlap > 0 && e < n {
            w[chunk - overlap..].copy_from_slice(&linspace(1.0, 0.0, overlap));
        }
        acc.resize(e - base, 0.0);
        wsum.resize(e - base, 0.0);
        for j in 0..chunk {
            acc[s - base + j] += y[j] * w[j];
            wsum[s - base + j] += w[j];
        }
        let upto = if i + 1 < starts.len() { starts[i + 1] } else { n };
        let cnt = upto - base;
        let done: Vec<f32> = (0..cnt).map(|j| acc[j] / wsum[j].max(1e-6)).collect();
        write(&done)?;
        acc.drain(..cnt);
        wsum.drain(..cnt);
        base = upto;
        tick(upto);
    }
    Ok(())
}

/// Denoises interleaved f32 audio from `input` into interleaved f32 at `output`.
/// `tick` receives the number of per-channel samples completed (summed over passes).
pub fn denoise_raw(
    model: &dyn Denoiser,
    input: &Path,
    channels: usize,
    output: &Path,
    downmix: bool,
    mut tick: impl FnMut(usize),
) -> Result<usize> {
    let n = std::fs::metadata(input)?.len() as usize / 4 / channels;
    if n == 0 {
        anyhow::bail!("the audio stream is empty");
    }
    let mut src = File::open(input)?;
    let passes: Vec<Option<usize>> = if downmix { vec![None] } else { (0..channels).map(Some).collect() };
    let dir = output.parent().unwrap_or(Path::new("."));
    let mut parts = Vec::new();
    let mut done_before = 0usize;
    for (pi, ch) in passes.iter().enumerate() {
        let part = dir.join(format!("part{pi}.raw"));
        let mut w = BufWriter::new(File::create(&part)?);
        enhance_stream(
            model,
            n,
            |s, l| read_channel(&mut src, channels, s, l, *ch),
            |xs| write_f32(&mut w, xs),
            |upto| tick(done_before + upto),
        )?;
        w.flush()?;
        done_before += n;
        parts.push(part);
    }
    // Interleave the per-channel results.
    let mut readers: Vec<File> = parts.iter().map(|p| File::open(p)).collect::<std::io::Result<_>>()?;
    let mut out = BufWriter::new(File::create(output)?);
    const BLOCK: usize = 1 << 18;
    let mut left = n;
    while left > 0 {
        let len = BLOCK.min(left);
        let mut bufs = Vec::new();
        for r in readers.iter_mut() {
            let mut b = vec![0u8; len * 4];
            r.read_exact(&mut b)?;
            bufs.push(b);
        }
        let mut bytes = Vec::with_capacity(len * channels * 4);
        for i in 0..len {
            for c in 0..channels {
                let b = if downmix { &bufs[0] } else { &bufs[c] };
                bytes.extend_from_slice(&b[i * 4..i * 4 + 4]);
            }
        }
        out.write_all(&bytes)?;
        left -= len;
    }
    out.flush()?;
    for p in parts {
        let _ = std::fs::remove_file(p);
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{test_config, Config};

    /// A "model" that returns its input: any chunking or crossfade error shows up in the output.
    struct Identity(Config);
    impl Denoiser for Identity {
        fn cfg(&self) -> &Config {
            &self.0
        }
        fn run(&self, audio: &[f32]) -> Result<Vec<f32>> {
            assert_eq!(audio.len(), self.0.chunk_samples().min(audio.len()));
            Ok(audio.to_vec())
        }
    }

    fn roundtrip(n: usize, channels: usize, downmix: bool) {
        // sample rate 1000: 1 s chunks, 0.25 s overlap
        let model = Identity(test_config(1000, 1.0, 0.25));
        let dir = tempfile::tempdir().unwrap();
        let (input, output) = (dir.path().join("in.raw"), dir.path().join("out.raw"));
        let mut bytes = Vec::new();
        let sample = |i: usize, c: usize| ((i * 31 + c * 977) % 1000) as f32 / 1000.0 - 0.5;
        for i in 0..n {
            for c in 0..channels {
                bytes.extend_from_slice(&sample(i, c).to_le_bytes());
            }
        }
        std::fs::write(&input, bytes).unwrap();
        let done = denoise_raw(&model, &input, channels, &output, downmix, |_| {}).unwrap();
        assert_eq!(done, n);
        let out = std::fs::read(&output).unwrap();
        assert_eq!(out.len(), n * channels * 4);
        for i in 0..n {
            for c in 0..channels {
                let got = f32::from_le_bytes(out[(i * channels + c) * 4..(i * channels + c) * 4 + 4].try_into().unwrap());
                let want = if downmix { (0..channels).map(|k| sample(i, k)).sum::<f32>() / channels as f32 } else { sample(i, c) };
                assert!((got - want).abs() < 1e-6, "n={n} ch={channels} sample {i}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn short_clip_is_one_pass() {
        roundtrip(700, 1, false);
        roundtrip(1000, 1, false);
    }

    #[test]
    fn long_clips_crossfade_seamlessly() {
        for n in [1001, 1750, 2000, 2999, 7123] {
            roundtrip(n, 1, false);
        }
    }

    #[test]
    fn multichannel_and_downmix() {
        roundtrip(3333, 2, false);
        roundtrip(3333, 3, true);
    }
}
