import os
import time
from pathlib import Path

from textplease.gradio_worker import PersistentPipelineWorker


def _record_process(config: dict) -> None:
    time.sleep(config.get("delay", 0))
    with Path(config["output_path"]).open("a") as output_file:
        output_file.write(f"{os.getpid()}\n")


def _write_temporary_audio(config: dict) -> None:
    Path(config["_temporary_directory"], "audio.wav").write_bytes(b"sensitive audio")
    Path(config["_temporary_directory"], "transcript.tsv").write_text("partial transcript")
    time.sleep(10)


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


def test_forced_termination_removes_job_temporary_directory(tmp_path):
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"original audio")
    output_path = tmp_path / "output.csv"
    output_path.write_text("existing transcript")
    worker = PersistentPipelineWorker(_write_temporary_audio)

    try:
        job_id = worker.submit(
            {
                "input_path": str(input_path),
                "model_name": "test-model",
                "device": "cpu",
                "output_path": str(output_path),
            },
            str(tmp_path / "worker.log"),
        )
        deadline = time.monotonic() + 10
        temporary_audio = []
        while time.monotonic() < deadline:
            temporary_audio = list(tmp_path.glob(".textplease-*/audio.wav"))
            if temporary_audio:
                break
            time.sleep(0.01)

        temporary_directories = list(tmp_path.glob(".textplease-*"))
        assert temporary_audio
        assert len(temporary_directories) == 1
        if os.name != "nt":
            assert temporary_directories[0].stat().st_mode & 0o777 == 0o700
        worker.terminate(force=True)

        assert worker.result(job_id)[0]
        assert not temporary_directories[0].exists()
        assert input_path.read_bytes() == b"original audio"
        assert output_path.read_text() == "existing transcript"
    finally:
        worker.shutdown()
