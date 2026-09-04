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

    def analyze(self, audio: np.ndarray) -> dict:
        rms = self.rms(audio)
        peak = self.peak(audio)

        return {
            "rms": rms,
            "rms_db": self.dbfs(rms),
            "peak": peak,
            "peak_db": self.dbfs(peak),
        }