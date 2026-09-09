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
    def clipping_count(audio: np.ndarray, threshold: float = 0.999) -> int:
        audio = audio.astype(np.float32).flatten()

        return int(np.sum(np.abs(audio) >= threshold))


    @staticmethod
    def noise_floor(audio: np.ndarray) -> float:
        audio = audio.astype(np.float32).flatten()

        if len(audio) == 0:
            return -100.0

        rms = np.sqrt(np.mean(np.square(audio)))

        if rms <= 1e-10:
            return -100.0

        return float(20 * np.log10(rms))

    @staticmethod
    def calculate_noise_floor(frames: list[np.ndarray]) -> float:
        if not frames:
            return -100.0

        rms_values = []

        for audio in frames:
            audio = audio.astype(np.float32).flatten()

            if len(audio) == 0:
                continue

            rms = np.sqrt(np.mean(np.square(audio)))

            if rms > 1e-10:
                rms_values.append(rms)

        if not rms_values:
            return -100.0

        # Exclude excessively high frames from the noise floor calculation.
        rms_values = np.array(rms_values, dtype=np.float32)

        # Create a baseline from the bottom 80% segment.
        threshold = np.percentile(rms_values, 80)
        quiet_frames = rms_values[rms_values <= threshold]

        if len(quiet_frames) == 0:
            quiet_frames = rms_values

        noise_rms = float(np.median(quiet_frames))

        return AudioAnalyzer.dbfs(noise_rms)

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
            "clipping_count" : self.clipping_count(audio),
            "noise_floor" : self.noise_floor(audio)
        }