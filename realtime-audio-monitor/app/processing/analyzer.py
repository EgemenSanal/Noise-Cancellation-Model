import numpy as np


class AudioAnalyzer:
    @staticmethod
    def rms(audio: np.ndarray) -> float:
        audio = audio.astype(np.float32)

        return float(np.sqrt(np.mean(np.square(audio))))

    @staticmethod
    def peak(audio: np.ndarray) -> float:
        return float(np.max(np.abs(audio)))

    @staticmethod
    def dbfs(value: float) -> float:
        if value <= 1e-10:
            return -100.0

        return float(20 * np.log10(value))

    @staticmethod
    def fft(audio: np.ndarray, sample_rate: int):
        audio = audio.astype(np.float32).flatten()

        window = np.hanning(len(audio))
        windowed = audio * window

        spectrum = np.fft.rfft(windowed)
        magnitude = np.abs(spectrum)

        frequencies = np.fft.rfftfreq(
            len(audio),
            d=1.0 / sample_rate,
        )

        magnitude_db = 20 * np.log10(
            np.maximum(magnitude, 1e-10)
        )

        return frequencies, magnitude_db

    def analyze(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> dict:
        audio = audio.astype(np.float32).flatten()

        rms = self.rms(audio)
        peak = self.peak(audio)

        frequencies, spectrum_db = self.fft(
            audio,
            sample_rate,
        )

        return {
            "rms": rms,
            "rms_db": self.dbfs(rms),
            "peak": peak,
            "peak_db": self.dbfs(peak),
            "waveform": audio,
            "frequencies": frequencies,
            "spectrum_db": spectrum_db,
        }
