# On-device (mobile) deployment — ONNX streaming onset detector

This describes everything an app needs to run the piano onset detector on-device from
a microphone, with **bit-identical** output to the PyTorch/offline pipeline (verified by
`test_onnx_parity.py`: numpy mel == torchaudio mel to 1e-6; ONNX streaming == PyTorch
streaming == whole-piece offline inference, IoU 1.0).

Reference implementation: **`onnx_runtime.py`** (pure numpy + onnxruntime, no torch).
Port that file's logic to Swift/Kotlin/C++; this doc is the spec, that file is the proof.

## Artifacts (in `output/`, produced by `export_onnx.py`)

| file | what |
|---|---|
| `mel229_fixed_aug_2x_chunk16.onnx` | **steady-state** graph — every chunk after the first |
| `mel229_fixed_aug_2x_chunk16_first.onnx` | **first-chunk** graph — used exactly once at stream start |
| `mel229_fixed_aug_2x_chunk16.json` | all config below (shapes, preproc, decode) |
| `mel229_fixed_aug_2x_melfb.npy` | `[1025, 229]` float32 mel filterbank (power-spectrum → mel) |

Two graphs share the same weights but differ only in input length / edge handling. The
first chunk has no left context (conv-internal edge padding); reproducing it exactly is
what makes streaming match whole-piece inference — do not approximate it with zero-padding.

Regenerate for any checkpoint: `uv run python export_onnx.py --model <ckpt>.pt`

## 1. Audio capture

- **16 kHz, mono, float32** in roughly `[-1, 1]`. Capture at 16 kHz directly — there is no
  resampling in the graph. (44.1/48 kHz mic → resample to 16 kHz before the pipeline.)
- Maintain a rolling audio buffer; you need ~250 ms of audio before the first detection.

## 2. Log-mel features (per frame)  — see `compute_logmel()`

Compute a log-mel spectrogram identical to training. Params (from the JSON `preproc`):

| param | value |
|---|---|
| `sample_rate` | 16000 |
| `n_fft` / `win_length` | 2048 |
| `hop` | 160  (= 10 ms/frame) |
| `n_mels` | 229 |
| `f_min` / `f_max` | 20.0 / 8000.0 |
| window | **Hann, periodic** (`0.5 - 0.5*cos(2πn/N)`, n=0..N-1, N=2048) |
| `center` | true (reflect-pad the signal by `n_fft/2` before framing) |
| `power` | 2.0 (use `|STFT|²`, the power spectrum) |

Per frame:
1. STFT: reflect-pad signal by 1024, frame with hop=160 / win=2048, apply Hann, `rfft` → 1025 complex bins.
2. Power: `re² + im²` → `[1025]`.
3. Mel: `power @ filterbank` (filterbank is `[1025, 229]` from the `.npy`) → `[229]`.
4. Log: `log1p(x)` = `ln(1 + x)` (natural log).
5. Normalize (**fixed**, not per-frame): `(x - mel_mean) / mel_std` with
   `mel_mean = 0.50172`, `mel_std = 1.11801`.

Result: one `[229]` log-mel vector per 10 ms hop. The app computes these incrementally over
its rolling audio buffer; steady-state frame values are identical to a whole-signal STFT
(they differ only within `n_fft/2` of a buffer edge, which the rolling buffer keeps as context).

## 3. Streaming model loop (GRU state + two graphs)  — see `OnnxStreamingDetector`

State you carry between calls: the GRU hidden state `h` (`float32 [1, 1, 192]`), the rolling
mel buffer, and `fp` = number of mel frames already fed to the GRU.

Constants: `chunk = 16`, `ctx = 3`.

```
reset: buffer = [], fp = 0, h = zeros[1,1,192], first_done = false

on new mel frames -> append to buffer; then while buffer.len >= fp + chunk + ctx:
    win_start = max(0, fp - ctx)
    win_end   = fp + chunk + ctx
    window    = buffer[win_start : win_end]          # 19 frames first time, 22 after
    graph     = first_done ? steady : first
    probs[1,16,88], h = graph.run(mel_window = window[None], h_in = h)
    first_done = true
    fp += chunk
    feed probs[0]  (shape [16, 88]) to the peak picker (section 4)
  # trim consumed history, keep ctx frames of left context:
  keep = max(0, fp - ctx); buffer = buffer[keep:]; fp -= keep
```

- **First call** uses the `first` graph (19-frame window, no left context), `h_in = 0`.
  It outputs onset probs for mel frames `0..15`.
- **Every later call** uses the `steady` graph (22-frame window = 3 left-ctx + 16 + 3 right-ctx).
- `probs` are already sigmoid-applied onset probabilities in `[0,1]`, shape `[16, 88]`
  (16 frames × 88 piano keys).

## 4. Peak-picking (onset decisions)  — see `StreamingPeakPicker`

Per pitch, a frame `t` is an onset iff:

```
p[t] >= threshold  AND  p[t] > p[t-1]  AND  p[t] >= p[t+1]
```

plus a **per-pitch refractory** of `refractory_frames = 5` (don't emit the same pitch again
within 5 frames). Because the rule needs `p[t+1]`, hold back the last frame of each chunk and
decide it when the next chunk arrives (1-frame look-ahead). The picker is stateful across
chunks: keep the last 2 probability frames per call, and a per-pitch "last emitted frame".

`threshold` is the main tuning knob (see §6). `0.65` is the bundled default.

## 5. Output mapping

A picked event is `(frame_index, pitch_index)`:
- **time (seconds)** = `frame_index * hop / sample_rate` = `frame_index * 160 / 16000` = `frame_index * 0.01`
- **MIDI note** = `pitch_index + midi_low` = `pitch_index + 21`  (A0 = 21; pitch_index 0..87)

Frame indices are true global mel-frame indices: the first chunk covers frames `0..15`, so
times are correct from `t = 0` with no offset.

## 6. Latency & threshold

- **Latency** = `(chunk + ctx) * 10 ms` = **190 ms** audio-in → event-out (16 chunk frames +
  3 right-context frames). This is inherent to the conv-context streaming design.
- **Threshold** trade-off (note-level F1 on MAESTRO test, this model):

  | threshold | F1 | precision | recall | character |
  |---|---|---|---|---|
  | 0.40 | 0.890 | 0.851 | **0.934** | aggressive — catches most onsets, more false positives |
  | 0.50 | 0.905 | 0.888 | 0.924 | high-recall |
  | **0.65** | 0.918 | 0.932 | 0.906 | bundled default |
  | 0.70 | **0.920** | 0.944 | 0.898 | best F1 |
  | 0.80 | 0.918 | 0.966 | 0.877 | conservative |

  For a **teaching app** that filters detections against an expected score, prefer a **lower
  threshold (0.40–0.50)**: you can discard false positives cheaply against the score, but a
  missed onset can't be recovered. Real-world (home piano, OOD) precision is lower than these
  MAESTRO numbers, so tune on-device.

## 7. Sanity-check on device

Feed a known WAV (or a few seconds of a recording you've also run through
`uv run python stream.py --wav x.wav --model output/mel229_fixed_aug_2x.pt`) and confirm the
app reports the same pitches at the same times. `test_onnx_parity.py` is the host-side proof
that the ONNX path matches PyTorch exactly; this is the on-device equivalent.
