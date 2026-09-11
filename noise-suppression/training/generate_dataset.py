from pathlib import Path

from src.audio.mixer import create_noisy_sample


CLEAN_DIR = Path("dataset/clean")
NOISE_DIR = Path("dataset/noises")
OUTPUT_DIR = Path("dataset/noisy")

SNR_LEVELS = [
    -5,
    0,
    5,
    10,
    15,
]


def main():
    clean_files = sorted(CLEAN_DIR.glob("*.wav"))
    noise_files = sorted(NOISE_DIR.glob("*.wav"))

    if not clean_files:
        raise RuntimeError(
            "No clean WAV files found in dataset/clean"
        )

    if not noise_files:
        raise RuntimeError(
            "No noise WAV files found in dataset/noises"
        )

    generated = 0

    for clean_path in clean_files:
        for noise_path in noise_files:
            for snr_db in SNR_LEVELS:
                output_name = (
                    f"{clean_path.stem}"
                    f"__{noise_path.stem}"
                    f"__snr{snr_db}.wav"
                )

                output_path = OUTPUT_DIR / output_name

                create_noisy_sample(
                    clean_path=clean_path,
                    noise_path=noise_path,
                    output_path=output_path,
                    snr_db=snr_db,
                )

                generated += 1

                print(
                    f"Generated: {output_path}"
                )

    print()
    print(
        f"Dataset generation complete: "
        f"{generated} samples"
    )


if __name__ == "__main__":
    main()
