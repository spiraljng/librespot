//! Sample rate conversion into the player's rate.
//!
//! Spotify streams arrive at [`SAMPLE_RATE`] and the playback pipeline is
//! built around it, but a local file can carry any rate. Converting here
//! lets those files play instead of being rejected, and keeps every
//! downstream stage (sink, mixer, position tracking) on one rate.
//!
//! It uses a polyphase windowed-sinc filter and computes only the taps
//! needed for each output sample.

use std::f64::consts::PI;

/// Input samples each output sample is made from. Sixty-four give a
/// passband flat to about 19 kHz and images under the window's floor.
const TAPS: usize = 64;

pub struct Resampler {
    up: usize,
    down: usize,
    channels: usize,
    /// `up` phases of `TAPS` coefficients each, normalised to unit gain.
    taps: Vec<f64>,
    /// Interleaved input frames still in reach: the tail of what came
    /// before, then whatever has not produced its outputs yet.
    input: Vec<f64>,
    /// The frame in `input` the next output sample sits on or just after.
    next: usize,
    /// How far past `next` the output sits, in steps of `1 / up`.
    phase: usize,
}

impl Resampler {
    /// `None` when the rates agree and nothing needs doing.
    pub fn new(from_hz: u32, to_hz: u32, channels: usize) -> Option<Self> {
        if from_hz == to_hz || from_hz == 0 || to_hz == 0 || channels == 0 {
            return None;
        }
        let divisor = gcd(from_hz, to_hz);
        let up = (to_hz / divisor) as usize;
        let down = (from_hz / divisor) as usize;
        let half = TAPS / 2;
        Some(Self {
            up,
            down,
            channels,
            taps: kernel(up, down),
            input: vec![0.0; (half - 1) * channels],
            next: half - 1,
            phase: 0,
        })
    }

    /// Converts a chunk of interleaved frames. The output is what the
    /// input so far allows; the last few frames wait for the next chunk.
    pub fn process(&mut self, samples: &[f64]) -> Vec<f64> {
        self.input.extend_from_slice(samples);
        let half = TAPS / 2;
        let frames = self.input.len() / self.channels;
        let expected = samples.len() * self.up / self.down + self.channels;
        let mut out = Vec::with_capacity(expected);
        while self.next + half < frames {
            let taps = &self.taps[self.phase * TAPS..(self.phase + 1) * TAPS];
            let start = (self.next + 1 - half) * self.channels;
            for channel in 0..self.channels {
                let sum: f64 = taps
                    .iter()
                    .enumerate()
                    .map(|(k, tap)| self.input[start + k * self.channels + channel] * tap)
                    .sum();
                out.push(sum);
            }
            let position = self.phase + self.down;
            self.next += position / self.up;
            self.phase = position % self.up;
        }
        // Keep only the frames the next output still reaches back to.
        let keep_from = (self.next + 1 - half).min(frames);
        self.input.drain(..keep_from * self.channels);
        self.next -= keep_from;
        out
    }
}

/// The taps for every phase: a sinc cut just under the lower of the two
/// Nyquist limits, under a Blackman window, each phase scaled to unit
/// gain so a steady level comes out at the level it went in.
fn kernel(up: usize, down: usize) -> Vec<f64> {
    let half = (TAPS / 2) as f64;
    let cutoff = 0.475 * (up as f64 / down as f64).min(1.0);
    let mut taps = Vec::with_capacity(up * TAPS);
    for phase in 0..up {
        let offset = phase as f64 / up as f64;
        let start = taps.len();
        for k in 0..TAPS {
            let u = offset + half - 1.0 - k as f64;
            let x = u / half;
            let window = 0.42 + 0.5 * (PI * x).cos() + 0.08 * (2.0 * PI * x).cos();
            let angle = 2.0 * PI * cutoff * u;
            let sinc = if u == 0.0 { 1.0 } else { angle.sin() / angle };
            taps.push(2.0 * cutoff * sinc * window);
        }
        let sum: f64 = taps[start..].iter().sum();
        for tap in &mut taps[start..] {
            *tap /= sum;
        }
    }
    taps
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stereo tone at half scale.
    fn tone(hz: f64, rate: u32, frames: usize) -> Vec<f64> {
        (0..frames)
            .flat_map(|i| {
                let sample = 0.5 * (2.0 * PI * hz * i as f64 / f64::from(rate)).sin();
                [sample, sample]
            })
            .collect()
    }

    /// Where output frame `n` sits on the input's clock, in input frames:
    /// the filter is symmetric about its output, so there is no delay.
    fn input_time(resampler: &Resampler, n: usize) -> f64 {
        n as f64 * resampler.down as f64 / resampler.up as f64
    }

    fn check(from: u32, to: u32, tolerance: f64) {
        let mut resampler = Resampler::new(from, to, 2).unwrap();
        let out = resampler.process(&tone(1000.0, from, from as usize));
        let frames = out.len() / 2;
        assert!(
            (frames as i64 - i64::from(to)).abs() < 100,
            "{frames} frames"
        );
        let mut worst = 0.0f64;
        for n in 200..frames - 200 {
            let t = input_time(&resampler, n) / f64::from(from);
            let ideal = 0.5 * (2.0 * PI * 1000.0 * t).sin();
            worst = worst.max((out[2 * n] - ideal).abs());
            assert_eq!(out[2 * n], out[2 * n + 1]);
        }
        assert!(worst < tolerance, "{from} to {to}: off by {worst}");
    }

    #[test]
    fn the_same_rate_needs_nothing() {
        assert!(Resampler::new(44100, 44100, 2).is_none());
    }

    #[test]
    fn a_tone_keeps_its_pitch_and_level_going_up() {
        check(48000, 44100, 0.005);
    }

    #[test]
    fn a_tone_keeps_its_pitch_and_level_going_down() {
        check(22050, 44100, 0.02);
    }

    #[test]
    fn chunking_makes_no_difference() {
        let input = tone(440.0, 48000, 20_000);
        let whole = Resampler::new(48000, 44100, 2).unwrap().process(&input);
        let mut chunked = Resampler::new(48000, 44100, 2).unwrap();
        let mut out = Vec::new();
        let mut at = 0;
        for size in [2, 14, 200, 3256, 6, 1000, 2, 8000].iter().cycle() {
            if at >= input.len() {
                break;
            }
            let end = (at + size).min(input.len());
            out.extend(chunked.process(&input[at..end]));
            at = end;
        }
        assert_eq!(out, whole);
    }
}
