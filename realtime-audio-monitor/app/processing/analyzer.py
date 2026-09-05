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
    def zero_crossing_rate(audio: np.ndarray) -> float:
        audio = audio.astype(np.float32).flatten()

        if len(audio) < 2:
            return 0.0

        crossings = np.sum(
            np.signbit(audio[:-1]) != np.signbit(audio[1:])
        )

        return float(crossings / (len(audio) - 1))

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

        return frequencies, magnitude_db, magnitude

    @staticmethod
    def spectral_centroid(
        frequencies: np.ndarray,
        magnitude: np.ndarray,
    ) -> float:
        total = np.sum(magnitude)

        if total <= 1e-10:
            return 0.0

        return float(
            np.sum(frequencies * magnitude) / total
        )

    def analyze(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> dict:
        audio = audio.astype(np.float32).flatten()

        rms = self.rms(audio)
        peak = self.peak(audio)

        frequencies, spectrum_db, magnitude = self.fft(
            audio,
            sample_rate,
        )

        centroid = self.spectral_centroid(
            frequencies,
            magnitude,
        )

        zcr = self.zero_crossing_rate(audio)

        return {
            "rms": rms,
            "rms_db": self.dbfs(rms),
            "peak": peak,
            "peak_db": self.dbfs(peak),
            "waveform": audio,
            "frequencies": frequencies,
            "spectrum_db": spectrum_db,
            "spectral_centroid": centroid,
            "zero_crossing_rate": zcr,
        }