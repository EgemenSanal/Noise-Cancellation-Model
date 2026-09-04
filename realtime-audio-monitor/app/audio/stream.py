import queue

import numpy as np
import sounddevice as sd


class AudioStream:
    def __init__(
        self,
        sample_rate: int = 48_000,
        channels: int = 1,
        block_duration_ms: int = 20,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = int(sample_rate * block_duration_ms / 1000)

        self.audio_queue = queue.Queue(maxsize=20)
        self.stream = None

    def _callback(self, indata, frames, time, status):
        if status:
            print(f"Audio status: {status}")

        audio = indata.copy()

        try:
            self.audio_queue.put_nowait(audio)
        except queue.Full:
            # If the GUI/processing side cannot keep up drop the new frame instead of holding onto the old one
            pass

    def start(self):
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.block_size,
            callback=self._callback,
        )

        self.stream.start()

        print("Audio stream started")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Channels: {self.channels}")
        print(f"Block size: {self.block_size} samples")

    def read(self, timeout=None):
        return self.audio_queue.get(timeout=timeout)

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        print("Audio stream stopped")