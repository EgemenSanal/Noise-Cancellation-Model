from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(
    path: str | Path,
    sample_rate: int = 48_000,
) -> tuple[np.ndarray, int]:
    audio, original_sample_rate = sf.read(
        path,
        dtype="float32",
    )

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if original_sample_rate != sample_rate:
        raise ValueError(
            f"Expected {sample_rate} Hz, "
            f"got {original_sample_rate} Hz: {path}"
        )

    return audio, original_sample_rate


def save_audio(
    path: str | Path,
    audio: np.ndarray,
    sample_rate: int = 48_000,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    audio = np.asarray(audio, dtype=np.float32)

    sf.write(
        path,
        audio,
        sample_rate,
    )
