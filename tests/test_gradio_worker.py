import os
import time
from pathlib import Path

from textplease.gradio_worker import PersistentPipelineWorker


def _record_process(config: dict) -> None:
    time.sleep(config.get("delay", 0))
    with Path(config["output_path"]).open("a") as output_file:
        output_file.write(f"{os.getpid()}\n")


def _wait_for_result(worker: PersistentPipelineWorker, job_id: int) -> str | None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        done, error = worker.result(job_id)
        if done:
            return error
        time.sleep(0.01)
    raise TimeoutError("Worker did not finish the test job")


def test_worker_reuses_process_after_success(tmp_path):
    output_path = tmp_path / "worker_pids.txt"
    config = {
        "model_name": "test-model",
        "device": "cpu",
        "output_path": str(output_path),
    }
    worker = PersistentPipelineWorker(_record_process)

    try:
        first_job = worker.submit(config, str(tmp_path / "first.log"))
        assert _wait_for_result(worker, first_job) is None

        second_job = worker.submit(config, str(tmp_path / "second.log"))
        assert _wait_for_result(worker, second_job) is None
    finally:
        worker.shutdown()

    assert len(set(output_path.read_text().splitlines())) == 1


def test_worker_restarts_after_termination(tmp_path):
    output_path = tmp_path / "worker_pids.txt"
    config = {
        "model_name": "test-model",
        "device": "cpu",
        "output_path": str(output_path),
        "delay": 10,
    }
    worker = PersistentPipelineWorker(_record_process)

    try:
        stopped_job = worker.submit(config, str(tmp_path / "stopped.log"))
        stopped_pid = worker.pid
        worker.terminate()
        assert worker.result(stopped_job)[0]

        config["delay"] = 0
        successful_job = worker.submit(config, str(tmp_path / "successful.log"))
        assert worker.pid != stopped_pid
        assert _wait_for_result(worker, successful_job) is None
    finally:
        worker.shutdown()
