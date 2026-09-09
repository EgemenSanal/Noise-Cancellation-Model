import queue

import numpy as np
import sounddevice as sd


class AudioStream:
    def __init__(
        self,
        sample_rate: int = 48_000,
        channels: int = 1,
        block_duration_ms: int = 20,
        device=None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_duration_ms = block_duration_ms
        self.block_size = int(sample_rate * block_duration_ms / 1000)
        self.device = device

        self.audio_queue = queue.Queue(maxsize=20)
        self.stream = None
        self.total_frames = 0
        self.dropped_frames = 0

    def _callback(self, indata, frames, time, status):
        if status:
            print(f"Audio status: {status}")

        audio = indata.copy()
        self.total_frames += 1;

        try:
            self.audio_queue.put_nowait(audio)
        except queue.Full:
            self.dropped_frames += 1
            # Prevent latency buildup in the real-time system.
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                self.audio_queue.put_nowait(audio)
            except queue.Full:
                pass

    def start(self):
        if self.stream is not None:
            return

        self.audio_queue = queue.Queue(maxsize=20)

        self.total_frames = 0
        self.dropped_frames = 0

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.block_size,
            device=self.device,
            callback=self._callback,
        )

        self.stream.start()

    def read(self):
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def get_buffer_size(self) -> int:
        return self.audio_queue.qsize()

    def get_stats(self) -> dict:
        return {
            "total_frames": self.total_frames,
            "dropped_frames": self.dropped_frames,
            "buffer_size": self.get_buffer_size(),
            "buffer_capacity": self.audio_queue.maxsize,
        }

    @staticmethod
    def get_input_devices():
        devices = sd.query_devices()

        return [
            {
                "index": index,
                "name": device["name"],
                "channels": device["max_input_channels"],
                "sample_rate": device["default_samplerate"],
            }
            for index, device in enumerate(devices)
            if device["max_input_channels"] > 0
        ]
