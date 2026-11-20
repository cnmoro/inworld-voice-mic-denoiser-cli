//! ffmpeg / ffprobe glue: probing, decoding audio to raw PCM and remuxing.

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Debug, Clone)]
pub struct AudioStream {
    /// Position among the file's audio streams (0:a:N).
    pub ordinal: usize,
    pub codec: String,
    pub channels: usize,
    pub sample_rate: u32,
    pub bit_rate: Option<u64>,
    pub start_time: f64,
}

pub struct Ffmpeg {
    pub ffmpeg: String,
    pub ffprobe: String,
    encoders: String,
}

fn exe_name(base: &str) -> String {
    if cfg!(windows) { format!("{base}.exe") } else { base.to_string() }
}

impl Ffmpeg {
    /// `hint` may be the ffmpeg binary or the directory containing ffmpeg and ffprobe.
    pub fn locate(hint: Option<&str>) -> Result<Self> {
        let (ffmpeg, ffprobe) = match hint {
            Some(h) => {
                let p = PathBuf::from(h);
                if p.is_dir() {
                    (p.join(exe_name("ffmpeg")).to_string_lossy().into_owned(), p.join(exe_name("ffprobe")).to_string_lossy().into_owned())
                } else {
                    let probe = p.with_file_name(exe_name("ffprobe"));
                    (h.to_string(), probe.to_string_lossy().into_owned())
                }
            }
            None => ("ffmpeg".to_string(), "ffprobe".to_string()),
        };
        let out = Command::new(&ffmpeg).args(["-hide_banner", "-encoders"]).output().with_context(|| {
            format!("could not run '{ffmpeg}'. Install ffmpeg (https://ffmpeg.org/download.html) and make sure it is on PATH, or pass --ffmpeg <path>")
        })?;
        if !out.status.success() {
            bail!("'{ffmpeg}' failed to run");
        }
        Ok(Self { ffmpeg, ffprobe, encoders: String::from_utf8_lossy(&out.stdout).into_owned() })
    }

    fn has_encoder(&self, name: &str) -> bool {
        self.encoders.lines().any(|l| l.split_whitespace().nth(1) == Some(name))
    }

    pub fn probe(&self, path: &Path) -> Result<Vec<AudioStream>> {
        let out = Command::new(&self.ffprobe)
            .args(["-v", "error", "-select_streams", "a", "-show_entries",
                "stream=codec_name,channels,sample_rate,bit_rate,start_time", "-of", "json"])
            .arg(path)
            .output()
            .with_context(|| format!("could not run '{}'", self.ffprobe))?;
        if !out.status.success() {
            bail!("ffprobe could not read {}: {}", path.display(), String::from_utf8_lossy(&out.stderr).trim());
        }
        let v: serde_json::Value = serde_json::from_slice(&out.stdout)?;
        let mut streams = Vec::new();
        for (i, s) in v["streams"].as_array().cloned().unwrap_or_default().iter().enumerate() {
            let num = |k: &str| s[k].as_str().and_then(|x| x.parse::<f64>().ok());
            streams.push(AudioStream {
                ordinal: i,
                codec: s["codec_name"].as_str().unwrap_or("").to_string(),
                channels: s["channels"].as_u64().unwrap_or(1).max(1) as usize,
                sample_rate: num("sample_rate").unwrap_or(48000.0) as u32,
                bit_rate: num("bit_rate").map(|b| b as u64).filter(|b| *b > 0),
                start_time: num("start_time").unwrap_or(0.0),
            });
        }
        Ok(streams)
    }

    /// Decodes one audio stream to interleaved little-endian f32 at 48 kHz.
    pub fn decode(&self, input: &Path, s: &AudioStream, raw: &Path, rate: u32) -> Result<()> {
        let out = Command::new(&self.ffmpeg)
            .args(["-v", "error", "-nostdin", "-y", "-i"])
            .arg(input)
            .args(["-map", &format!("0:a:{}", s.ordinal), "-vn", "-sn", "-dn", "-f", "f32le", "-acodec", "pcm_f32le",
                "-ar", &rate.to_string(), "-ac", &s.channels.to_string()])
            .arg(raw)
            .stdout(Stdio::null())
            .output()?;
        if !out.status.success() {
            bail!("ffmpeg failed to decode audio: {}", String::from_utf8_lossy(&out.stderr).trim());
        }
        Ok(())
    }

    /// Encoder arguments that recreate the original audio codec where possible.
    fn codec_args(&self, s: &AudioStream, j: usize) -> Vec<String> {
        let candidates: &[&str] = match s.codec.as_str() {
            "aac" => &["aac"],
            "mp3" => &["libmp3lame", "mp3"],
            "mp2" => &["mp2"],
            "opus" => &["libopus", "opus"],
            "vorbis" => &["libvorbis", "vorbis"],
            "flac" => &["flac"],
            "alac" => &["alac"],
            "ac3" => &["ac3"],
            "eac3" => &["eac3"],
            "wmav2" => &["wmav2"],
            c if c.starts_with("pcm_") => &[],
            _ => &[],
        };
        let mut name = candidates.iter().find(|c| self.has_encoder(c)).map(|c| c.to_string());
        if name.is_none() && s.codec.starts_with("pcm_") && self.has_encoder(&s.codec) {
            name = Some(s.codec.clone());
        }
        let Some(name) = name else { return vec![] };
        let mut args = vec![format!("-c:a:{j}"), name.clone()];
        if name == "vorbis" {
            args.extend(["-strict".into(), "-2".into()]);
        }
        let lossy = !s.codec.starts_with("pcm_") && !matches!(s.codec.as_str(), "flac" | "alac");
        if lossy {
            if let Some(b) = s.bit_rate {
                args.extend([format!("-b:a:{j}"), b.to_string()]);
            }
        }
        args
    }

    fn mux_command(&self, original: &Path, audio: &[(AudioStream, PathBuf)], dest: &Path, rate: u32, keep_codec: bool) -> Command {
        let mut cmd = Command::new(&self.ffmpeg);
        cmd.args(["-v", "error", "-nostdin", "-y", "-i"]).arg(original);
        for (s, raw) in audio {
            cmd.args(["-itsoffset", &format!("{}", s.start_time), "-f", "f32le", "-ar", &rate.to_string(),
                "-ac", &s.channels.to_string(), "-i"]).arg(raw);
        }
        cmd.args(["-map", "0", "-map", "-0:a", "-ignore_unknown"]);
        for k in 0..audio.len() {
            cmd.args(["-map", &format!("{}:a", k + 1)]);
        }
        cmd.args(["-c:v", "copy", "-c:s", "copy", "-c:d", "copy", "-c:t", "copy", "-map_metadata", "0", "-map_chapters", "0"]);
        for (j, (s, _)) in audio.iter().enumerate() {
            cmd.args([format!("-map_metadata:s:a:{j}"), format!("0:s:a:{}", s.ordinal)]);
            if keep_codec {
                cmd.args([format!("-ar:a:{j}"), s.sample_rate.to_string()]);
                cmd.args(self.codec_args(s, j));
            }
        }
        cmd.arg(dest);
        cmd
    }

    /// Writes `dest` from the untouched streams of `original` plus the denoised audio.
    /// First tries to recreate each audio stream's codec and bitrate; if the container
    /// refuses, falls back to the container's default audio codec.
    pub fn mux(&self, original: &Path, audio: &[(AudioStream, PathBuf)], dest: &Path, rate: u32) -> Result<()> {
        let first = self.mux_command(original, audio, dest, rate, true).stdout(Stdio::null()).output()?;
        if first.status.success() {
            return Ok(());
        }
        let second = self.mux_command(original, audio, dest, rate, false).stdout(Stdio::null()).output()?;
        if !second.status.success() {
            bail!(
                "ffmpeg failed to write {}:\n{}\n(first attempt: {})",
                dest.display(),
                String::from_utf8_lossy(&second.stderr).trim(),
                String::from_utf8_lossy(&first.stderr).trim()
            );
        }
        Ok(())
    }
}
