"""
Real-time Progress Tracking System
Provides WebSocket-based progress updates for long-running operations
"""
import logging
from typing import Optional, Callable, Dict
from datetime import datetime
import time

logger = logging.getLogger(__name__)

try:
    from flask_socketio import SocketIO, emit
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False


class ProgressTracker:
    """
    Track progress of long-running operations with real-time updates
    """

    def __init__(self, task_id: str, total_steps: int, socketio: Optional['SocketIO'] = None):
        """
        Initialize progress tracker

        Args:
            task_id: Unique task identifier
            total_steps: Total number of steps
            socketio: SocketIO instance for real-time updates
        """
        self.task_id = task_id
        self.total_steps = total_steps
        self.current_step = 0
        self.status = "starting"
        self.message = ""
        self.start_time = time.time()
        self.socketio = socketio
        self.cancelled = False

        logger.info(f"Progress tracker initialized for task: {task_id}")

    def update(self, step: int, message: str = "", status: str = "running"):
        """
        Update progress

        Args:
            step: Current step number
            message: Status message
            status: Status (running, completed, error)
        """
        self.current_step = step
        self.message = message
        self.status = status

        # Calculate progress percentage
        progress = (self.current_step / self.total_steps) * 100 if self.total_steps > 0 else 0

        # Calculate ETA
        elapsed = time.time() - self.start_time
        if self.current_step > 0:
            eta = (elapsed / self.current_step) * (self.total_steps - self.current_step)
        else:
            eta = 0

        # Prepare progress data
        progress_data = {
            "task_id": self.task_id,
            "step": self.current_step,
            "total_steps": self.total_steps,
            "progress": round(progress, 1),
            "message": message,
            "status": status,
            "elapsed": round(elapsed, 1),
            "eta": round(eta, 1)
        }

        # Emit via WebSocket if available
        if self.socketio and SOCKETIO_AVAILABLE:
            try:
                self.socketio.emit('progress_update', progress_data, namespace='/progress')
            except Exception as e:
                logger.warning(f"Failed to emit progress update: {e}")

        logger.info(f"Progress [{self.task_id}]: {progress:.1f}% - {message}")

    def increment(self, message: str = ""):
        """Increment progress by one step"""
        self.update(self.current_step + 1, message)

    def complete(self, message: str = "Completed"):
        """Mark task as completed"""
        self.update(self.total_steps, message, "completed")

    def error(self, message: str):
        """Mark task as errored"""
        self.update(self.current_step, message, "error")

    def cancel(self):
        """Cancel the task"""
        self.cancelled = True
        self.update(self.current_step, "Cancelled by user", "cancelled")

    def is_cancelled(self) -> bool:
        """Check if task is cancelled"""
        return self.cancelled


class ProgressManager:
    """
    Manage multiple progress trackers
    """

    def __init__(self, socketio: Optional['SocketIO'] = None):
        """Initialize progress manager"""
        self.trackers: Dict[str, ProgressTracker] = {}
        self.socketio = socketio
        logger.info("ProgressManager initialized")

    def create_tracker(self, task_id: str, total_steps: int) -> ProgressTracker:
        """
        Create a new progress tracker

        Args:
            task_id: Unique task identifier
            total_steps: Total number of steps

        Returns:
            ProgressTracker instance
        """
        tracker = ProgressTracker(task_id, total_steps, self.socketio)
        self.trackers[task_id] = tracker
        return tracker

    def get_tracker(self, task_id: str) -> Optional[ProgressTracker]:
        """Get a progress tracker by ID"""
        return self.trackers.get(task_id)

    def remove_tracker(self, task_id: str):
        """Remove a completed tracker"""
        if task_id in self.trackers:
            del self.trackers[task_id]

    def get_all_progress(self) -> Dict:
        """Get progress for all active tasks"""
        result = {}
        for task_id, tracker in self.trackers.items():
            result[task_id] = {
                "step": tracker.current_step,
                "total_steps": tracker.total_steps,
                "progress": (tracker.current_step / tracker.total_steps) * 100 if tracker.total_steps > 0 else 0,
                "message": tracker.message,
                "status": tracker.status
            }
        return result


# Global progress manager instance
_progress_manager: Optional[ProgressManager] = None


def get_progress_manager() -> ProgressManager:
    """Get the global progress manager instance"""
    global _progress_manager
    if _progress_manager is None:
        _progress_manager = ProgressManager()
    return _progress_manager


def init_progress_manager(socketio: 'SocketIO'):
    """Initialize the global progress manager with SocketIO"""
    global _progress_manager
    _progress_manager = ProgressManager(socketio)
    logger.info("Global progress manager initialized with SocketIO")


class DiffusionProgressCallback:
    """
    Progress callback for diffusion models (Stable Diffusion)
    Compatible with diffusers library callback format
    """

    def __init__(self, tracker: ProgressTracker, total_steps: int):
        """
        Initialize callback

        Args:
            tracker: Progress tracker instance
            total_steps: Total diffusion steps
        """
        self.tracker = tracker
        self.total_steps = total_steps
        self.current_step = 0

    def __call__(self, step: int, timestep: int, latents):
        """
        Callback function called by diffusion pipeline

        Args:
            step: Current step
            timestep: Current timestep
            latents: Current latent tensors
        """
        self.current_step = step

        # Update tracker
        message = f"Diffusion step {step}/{self.total_steps}"
        self.tracker.update(step, message)

        # Check if cancelled
        if self.tracker.is_cancelled():
            raise InterruptedError("Generation cancelled by user")

        return latents
