import logging
import multiprocessing
from threading import RLock
from collections.abc import Callable
from multiprocessing.connection import Connection

from textplease.pipeline import run_transcription_pipeline
from textplease.utils.logging_config import configure_logging


logger = logging.getLogger(__name__)


def _run_worker(
    command_connection: Connection,
    result_connection: Connection,
    runner: Callable[[dict], None],
) -> None:
    configure_logging()
    try:
        while True:
            command = command_connection.recv()
            job_id, config, log_path = command
            file_handler = logging.FileHandler(log_path)
            logging.getLogger().addHandler(file_handler)
            try:
                runner(config)
            except Exception as error:
                logger.exception("Transcription failed")
                result_connection.send((job_id, str(error)))
                return
            else:
                result_connection.send((job_id, None))
            finally:
                logging.getLogger().removeHandler(file_handler)
                file_handler.close()
    except EOFError:
        return
    finally:
        command_connection.close()
        result_connection.close()


class PersistentPipelineWorker:
    """Run transcription jobs in one accelerator-safe child process."""

    def __init__(self, runner: Callable[[dict], None] = run_transcription_pipeline):
        """Initialise a worker without starting its process."""
        self._runner = runner
        self._context = multiprocessing.get_context("spawn")
        self._lock = RLock()
        self._process = None
        self._command_connection = None
        self._result_connection = None
        self._current_job_id: int | None = None
        self._next_job_id = 1
        self._results: dict[int, str | None] = {}
        self._model_key: tuple[str, str, str] | None = None
        self._last_exitcode: int | None = None

    @property
    def pid(self) -> int | None:
        """Return the worker process ID when it is running."""
        with self._lock:
            if self._process is None or not self._process.is_alive():
                return None
            return self._process.pid

    @property
    def exitcode(self) -> int | None:
        """Return the most recent worker exit code."""
        with self._lock:
            if self._process is not None and not self._process.is_alive():
                return self._process.exitcode
            return self._last_exitcode

    def submit(self, config: dict, log_path: str) -> int:
        """Submit one job, reusing a compatible idle worker process."""
        with self._lock:
            if self._current_job_id is not None:
                done, error = self._result(self._current_job_id)
                if not done:
                    raise RuntimeError("Another transcription is already running")
                if error is not None:
                    self._stop_worker()

            model_key = (
                config["model_name"],
                config.get("device", "cpu"),
                config.get("embedding_model", "all-MiniLM-L6-v2"),
            )
            if self._process is not None and self._process.is_alive() and model_key != self._model_key:
                self._stop_worker()
            if self._process is None or not self._process.is_alive():
                self._start_worker()

            job_id = self._next_job_id
            self._next_job_id += 1
            self._current_job_id = job_id
            self._model_key = model_key
            if self._command_connection is None:
                raise RuntimeError("Worker command connection is unavailable")
            self._command_connection.send((job_id, config, log_path))
            return job_id

    def result(self, job_id: int) -> tuple[bool, str | None]:
        """Return whether a job finished and its error message, if any."""
        with self._lock:
            return self._result(job_id)

    def is_running(self, job_id: int) -> bool:
        """Return whether the requested job is currently running."""
        done, _ = self.result(job_id)
        return not done

    def terminate(self, force: bool = False) -> None:
        """Stop the worker, discarding its loaded models."""
        with self._lock:
            self._stop_worker(force=force)

    def shutdown(self) -> None:
        """Release the worker process and its model memory."""
        self.terminate()

    def _result(self, job_id: int) -> tuple[bool, str | None]:
        self._drain_results()
        if job_id in self._results:
            return True, self._results[job_id]
        if job_id != self._current_job_id:
            return True, "Unknown transcription job"
        if self._process is None or not self._process.is_alive():
            exitcode = self.exitcode
            error = f"Worker exited with code {exitcode}"
            self._results[job_id] = error
            return True, error
        return False, None

    def _drain_results(self) -> None:
        if self._result_connection is None:
            return
        try:
            while self._result_connection.poll():
                job_id, error = self._result_connection.recv()
                self._results[job_id] = error
        except (EOFError, OSError):
            return

    def _start_worker(self) -> None:
        child_commands, parent_commands = self._context.Pipe(duplex=False)
        parent_results, child_results = self._context.Pipe(duplex=False)
        process = self._context.Process(
            target=_run_worker,
            args=(child_commands, child_results, self._runner),
            daemon=True,
        )
        process.start()
        child_commands.close()
        child_results.close()
        self._process = process
        self._command_connection = parent_commands
        self._result_connection = parent_results
        self._last_exitcode = None

    def _stop_worker(self, force: bool = False) -> None:
        if self._process is None:
            return
        if self._process.is_alive():
            if force:
                self._process.kill()
            else:
                self._process.terminate()
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=5)

        self._last_exitcode = self._process.exitcode
        if self._current_job_id is not None and self._current_job_id not in self._results:
            self._results[self._current_job_id] = f"Worker exited with code {self._last_exitcode}"
        self._process.close()
        self._process = None
        if self._command_connection is not None:
            self._command_connection.close()
            self._command_connection = None
        if self._result_connection is not None:
            self._result_connection.close()
            self._result_connection = None
        self._model_key = None
