"""
Purpose: Thread-safe Video Ingestion Reader using Producer-Consumer pattern.
Dependencies: cv2, queue, threading, time, shared.logging
Inputs: Source path/URL, target FPS, queue size
Outputs: Iterator/Generator yielding FrameData objects
"""

import queue
import threading
import time
from typing import Generator, Optional
import cv2
from services.ingestion.frame import FrameData
from services.ingestion.metadata import VideoMetadata, extract_video_metadata
from shared.logging import setup_logger

logger = setup_logger("ingestion_reader", service_name="ingestion")


class VideoIngestionReader:
    """Thread-safe background video reader reading frames into a bounded queue buffer."""

    def __init__(
        self,
        source: str,
        target_fps: Optional[float] = None,
        max_queue_size: int = 100,
    ):
        self.source = source
        self.metadata: VideoMetadata = extract_video_metadata(source)
        self.target_fps = target_fps or self.metadata.fps
        self.max_queue_size = max_queue_size

        # Calculate frame stride for downsampling
        self.stride = max(1, int(round(self.metadata.fps / self.target_fps)))

        self._frame_queue: queue.Queue[Optional[FrameData]] = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Starts background frame reader thread."""
        if self._worker_thread and self._worker_thread.is_alive():
            logger.warning("VideoIngestionReader worker thread is already running.")
            return

        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._worker_thread.start()
        logger.info(
            f"Started VideoIngestionReader thread for source: {self.source} "
            f"(Source FPS: {self.metadata.fps}, Target FPS: {self.target_fps}, Stride: {self.stride})"
        )

    def _reader_loop(self) -> None:
        """Background thread worker fetching frames from OpenCV VideoCapture."""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            logger.error(f"Failed to open video source in worker thread: {self.source}")
            self._frame_queue.put(None)
            return

        raw_frame_idx = 0
        emitted_frame_idx = 0

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.info("Video reach end-of-file or stream disconnected.")
                    break

                raw_frame_idx += 1

                # Apply frame stride downsampling
                if raw_frame_idx % self.stride != 0:
                    continue

                timestamp_seconds = round(raw_frame_idx / self.metadata.fps, 3)
                frame_data = FrameData(
                    frame_number=raw_frame_idx,
                    timestamp_seconds=timestamp_seconds,
                    image=frame,
                    metadata=self.metadata,
                )

                # Block until queue space is available or stop event is set
                while not self._stop_event.is_set():
                    try:
                        self._frame_queue.put(frame_data, timeout=0.1)
                        emitted_frame_idx += 1
                        break
                    except queue.Full:
                        continue

        except Exception as e:
            logger.error(f"Error in reader thread: {e}", exc_info=True)
        finally:
            cap.release()
            # Push sentinel None value indicating stream completion
            self._frame_queue.put(None)
            logger.info(f"Reader worker thread exiting. Total emitted frames: {emitted_frame_idx}")

    def stream_frames(self) -> Generator[FrameData, None, None]:
        """Generator producing FrameData objects from the bounded queue buffer."""
        if not self._worker_thread or not self._worker_thread.is_alive():
            self.start()

        while True:
            try:
                item = self._frame_queue.get(timeout=1.0)
                if item is None:
                    # Received sentinel signal -> Stream ended
                    break
                yield item
                self._frame_queue.task_done()
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue

    def stop(self) -> None:
        """Signals worker thread to stop and releases resources."""
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        logger.info("VideoIngestionReader stopped successfully.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
