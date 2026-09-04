import time

from app.audio.stream import AudioStream
from app.processing.analyzer import AudioAnalyzer


def main():
    stream = AudioStream(
        sample_rate=48_000,
        channels=1,
        block_duration_ms=20,
    )

    analyzer = AudioAnalyzer()

    try:
        stream.start()

        while True:
            audio = stream.read(timeout=1)

            result = analyzer.analyze(audio)

            print(
                f"RMS: {result['rms_db']:>7.2f} dBFS | "
                f"Peak: {result['peak_db']:>7.2f} dBFS"
            )

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        stream.stop()


if __name__ == "__main__":
    main()