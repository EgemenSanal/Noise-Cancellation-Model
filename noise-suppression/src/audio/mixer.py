from pathlib import Path

import numpy as np
import soundfile as sf


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio))))


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))

    if peak <= 1e-10:
        return audio.astype(np.float32)

    return (audio / peak).astype(np.float32)


def match_length(
    noise: np.ndarray,
    target_length: int,
) -> np.ndarray:
    if len(noise) < target_length:
        repeats = int(np.ceil(target_length / len(noise)))
        noise = np.tile(noise, repeats)

    start = np.random.randint(
        0,
        len(noise) - target_length + 1,
    )

    return noise[start:start + target_length]


def mix_at_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    clean = clean.astype(np.float32)
    noise = noise.astype(np.float32)

    clean_rms = rms(clean)
    noise_rms = rms(noise)

    if clean_rms <= 1e-10 or noise_rms <= 1e-10:
        return clean.copy()

    desired_noise_rms = (
        clean_rms / (10 ** (snr_db / 20))
    )

    noise = noise * (
        desired_noise_rms / noise_rms
    )

    mixed = clean + noise

    peak = np.max(np.abs(mixed))

    if peak > 0.99:
        mixed = mixed * (0.99 / peak)

    return mixed.astype(np.float32)


def create_noisy_sample(
    clean_path: str | Path,
    noise_path: str | Path,
    output_path: str | Path,
    snr_db: float,
    sample_rate: int = 48_000,
) -> None:
    clean, clean_rate = sf.read(
        clean_path,
        dtype="float32",
    )

    noise, noise_rate = sf.read(
        noise_path,
        dtype="float32",
    )

    if clean_rate != sample_rate:
        raise ValueError(
            f"Clean audio must be {sample_rate} Hz"
        )

    if noise_rate != sample_rate:
        raise ValueError(
            f"Noise audio must be {sample_rate} Hz"
        )

    if clean.ndim > 1:
        clean = np.mean(clean, axis=1)

    if noise.ndim > 1:
        noise = np.mean(noise, axis=1)

    clean = normalize_audio(clean)
    noise = match_length(noise, len(clean))

    noisy = mix_at_snr(
        clean,
        noise,
        snr_db,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sf.write(
        output_path,
        noisy,
        sample_rate,
    )
