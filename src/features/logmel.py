"""Log-mel and AST filter-bank front-ends operating on 16 kHz, 4 s waveforms."""
import librosa
import numpy as np
import torch
import torchaudio.compliance.kaldi as ta_kaldi

from . import params as P


def n_valid_frames(n_samples: int) -> int:
    """Frames whose centre lies inside the real (unpadded) signal, with center=True."""
    return min(P.N_FRAMES, (max(n_samples, 1) - 1) // P.HOP + 1)


def logmel_db(y: np.ndarray) -> np.ndarray:
    """Power log-mel in dB re 1.0, shape (N_MELS, N_FRAMES), NO floor clamp.

    Digital silence (zero padding) maps to 10*log10(AMIN) = -100 dB. Clamping to a
    sensible floor is a separate, documented step (`apply_floor`), so the raw cache can
    be used to study the choice.
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=P.SR, n_fft=P.N_FFT, hop_length=P.HOP, n_mels=P.N_MELS,
        fmin=P.FMIN, fmax=P.FMAX, power=2.0, center=True,
    )
    return librosa.power_to_db(S, ref=1.0, amin=P.AMIN, top_db=None).astype(np.float32)


def apply_floor(x, floor_db: float):
    """Clamp log-mel values from below. Works on numpy arrays and torch tensors."""
    if isinstance(x, torch.Tensor):
        return x.clamp(min=floor_db)
    return np.maximum(x, floor_db)


def ast_fbank(y: np.ndarray) -> np.ndarray:
    """Un-normalised AST fbank, shape (AST_FRAMES, AST_MEL_BINS).

    Same call as transformers.ASTFeatureExtractor: Kaldi fbank (Hann window, HTK compat,
    10 ms shift, no dither), then zero-padded in fbank space to AST_FRAMES. Normalisation
    with the checkpoint constants is applied at load time (`ast_normalize`).
    """
    wav = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)).unsqueeze(0)
    fb = ta_kaldi.fbank(
        wav, sample_frequency=P.SR, window_type="hanning", num_mel_bins=P.AST_MEL_BINS,
        htk_compat=True, use_energy=False, dither=0.0, frame_shift=10,
    )
    pad = P.AST_FRAMES - fb.shape[0]
    if pad > 0:
        fb = torch.nn.functional.pad(fb, (0, 0, 0, pad))
    elif pad < 0:
        fb = fb[: P.AST_FRAMES]
    return fb.numpy().astype(np.float32)


def ast_normalize(fb):
    return (fb - P.AST_MEAN) / (P.AST_STD * 2)
