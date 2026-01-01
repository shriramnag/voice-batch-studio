# VoiceBatch Studio Pro - Audio Processing Module (Mel-Spectrogram)
from functools import lru_cache
from scipy import signal
import numpy as np
import librosa

@lru_cache()
def mel_basis(hp):
    """ऑडियो फ़्रीक्वेंसी को मेल स्केल में बदलने के लिए बेसिस तैयार करना"""
    assert hp.fmax <= hp.sample_rate // 2
    return librosa.filters.mel(
        sr=hp.sample_rate,
        n_fft=hp.n_fft,
        n_mels=hp.num_mels,
        fmin=hp.fmin,
        fmax=hp.fmax)

def preemphasis(wav, hp):
    """आवाज़ को साफ़ करने के लिए प्री-एम्फेसिस फ़िल्टर"""
    assert hp.preemphasis != 0
    wav = signal.lfilter([1, -hp.preemphasis], [1], wav)
    wav = np.clip(wav, -1, 1)
    return wav

def melspectrogram(wav, hp, pad=True):
    """मेन फंक्शन जो ऑडियो वेव को स्पेक्ट्रोग्राम में बदलता है"""
    if hp.preemphasis > 0:
        wav = preemphasis(wav, hp)
        assert np.abs(wav).max() - 1 < 1e-07

    spec_complex = _stft(wav, hp, pad=pad)
    spec_magnitudes = np.abs(spec_complex)

    if hp.mel_power != 1.0:
        spec_magnitudes **= hp.mel_power

    mel = np.dot(mel_basis(hp), spec_magnitudes)
    
    if hp.mel_type == "db":
        mel = _amp_to_db(mel, hp)

    if hp.normalized_mels:
        mel = _normalize(mel, hp).astype(np.float32)

    return mel 

def _stft(y, hp, pad=True):
    return librosa.stft(
        y,
        n_fft=hp.n_fft,
        hop_length=hp.hop_size,
        win_length=hp.win_size,
        center=pad,
        pad_mode="reflect",
    )

def _amp_to_db(x, hp):
    return 20 * np.log10(np.maximum(hp.stft_magnitude_min, x))

def _normalize(s, hp, headroom_db=15):
    min_level_db = 20 * np.log10(hp.stft_magnitude_min)
    s = (s - min_level_db) / (-min_level_db + headroom_db)
    return s
