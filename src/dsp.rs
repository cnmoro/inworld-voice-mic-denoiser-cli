//! STFT / iSTFT matching `torch.stft(center=True, pad_mode="reflect")` and `torch.istft`.

use rayon::prelude::*;
use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use std::sync::Arc;

pub struct Stft {
    pub n_fft: usize,
    pub hop: usize,
    pub bins: usize,
    window: Vec<f32>,
    fwd: Arc<dyn Fft<f32>>,
    inv: Arc<dyn Fft<f32>>,
}

impl Stft {
    pub fn new(n_fft: usize, hop: usize) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        // torch.hann_window is periodic by default.
        let window = (0..n_fft)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n_fft as f64).cos() as f32)
            .collect();
        Self {
            n_fft,
            hop,
            bins: n_fft / 2 + 1,
            window,
            fwd: planner.plan_fft_forward(n_fft),
            inv: planner.plan_fft_inverse(n_fft),
        }
    }

    pub fn num_frames(&self, samples: usize) -> usize {
        1 + samples / self.hop
    }

    /// Returns spectra laid out as [frames][bins][re, im].
    pub fn forward(&self, audio: &[f32]) -> Vec<f32> {
        let pad = self.n_fft / 2;
        let n = audio.len();
        let mut padded = Vec::with_capacity(n + 2 * pad);
        if n > pad {
            for i in (1..=pad).rev() {
                padded.push(audio[i]);
            }
            padded.extend_from_slice(audio);
            for i in 0..pad {
                padded.push(audio[n - 2 - i]);
            }
        } else {
            padded.resize(pad, 0.0);
            padded.extend_from_slice(audio);
            padded.resize(n + 2 * pad, 0.0);
        }
        let frames = self.num_frames(n);
        let mut out = vec![0f32; frames * self.bins * 2];
        out.par_chunks_mut(self.bins * 2)
            .enumerate()
            .for_each(|(f, dst)| {
                let start = f * self.hop;
                let mut buf: Vec<Complex32> = (0..self.n_fft)
                    .map(|i| Complex32::new(padded[start + i] * self.window[i], 0.0))
                    .collect();
                self.fwd.process(&mut buf);
                for (k, c) in buf.iter().take(self.bins).enumerate() {
                    dst[2 * k] = c.re;
                    dst[2 * k + 1] = c.im;
                }
            });
        out
    }

    /// Inverse of `forward`, trimmed to `length` samples.
    pub fn inverse(&self, spec: &[f32], length: usize) -> Vec<f32> {
        let frames = spec.len() / (self.bins * 2);
        let total = self.n_fft + self.hop * (frames - 1);
        let scale = 1.0 / self.n_fft as f32;
        let frames_td: Vec<Vec<f32>> = (0..frames)
            .into_par_iter()
            .map(|f| {
                let src = &spec[f * self.bins * 2..(f + 1) * self.bins * 2];
                let mut buf = vec![Complex32::new(0.0, 0.0); self.n_fft];
                for k in 0..self.bins {
                    let (re, mut im) = (src[2 * k], src[2 * k + 1]);
                    if k == 0 || k == self.n_fft / 2 {
                        im = 0.0;
                    }
                    buf[k] = Complex32::new(re, im);
                    if k > 0 && k < self.n_fft / 2 {
                        buf[self.n_fft - k] = Complex32::new(re, -im);
                    }
                }
                self.inv.process(&mut buf);
                (0..self.n_fft).map(|i| buf[i].re * scale * self.window[i]).collect()
            })
            .collect();
        let mut y = vec![0f32; total];
        let mut env = vec![0f32; total];
        for (f, seg) in frames_td.iter().enumerate() {
            let start = f * self.hop;
            for i in 0..self.n_fft {
                y[start + i] += seg[i];
                env[start + i] += self.window[i] * self.window[i];
            }
        }
        let off = self.n_fft / 2;
        (0..length)
            .map(|i| {
                let j = off + i;
                if j < total && env[j] > 1e-11 {
                    y[j] / env[j]
                } else {
                    0.0
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stft_roundtrip_is_identity() {
        let stft = Stft::new(2048, 480);
        let x: Vec<f32> = (0..20_000).map(|i| ((i as f32) * 0.013).sin() * 0.5 + ((i * 7 % 101) as f32 - 50.0) / 200.0).collect();
        let spec = stft.forward(&x);
        assert_eq!(spec.len(), stft.num_frames(x.len()) * stft.bins * 2);
        let y = stft.inverse(&spec, x.len());
        let err = x.iter().zip(&y).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(err < 1e-4, "round trip error {err}");
    }
}
