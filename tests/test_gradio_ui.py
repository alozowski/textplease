import os
from unittest.mock import Mock

import yaml
import pytest

from textplease import gradio_ui


def test_start_transcription_uses_gradio_cached_file(tmp_path):
    upload_path = tmp_path / "gradio-cache" / "recording.mp3"
    upload_path.parent.mkdir()
    upload_path.touch()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    worker = Mock(pid=1234)
    worker.submit.return_value = 7

    result = gradio_ui.start_transcription(
        worker,
        output_dir,
        str(upload_path),
        0.75,
        2.0,
        100,
        3,
        15,
        "en",
        "cpu",
        None,
    )

    config, log_path = worker.submit.call_args.args
    run = result[-1]
    workspace_path = run["workspace_path"]
    assert config["input_path"] == str(upload_path)
    assert workspace_path.parent == output_dir
    assert workspace_path.name.startswith("job-")
    assert log_path == str(workspace_path / "run.log")
    assert run["output_path"] == workspace_path / "recording_transcript.csv"
    assert run["config_path"] == workspace_path / "config.yaml"
    assert yaml.safe_load(run["config_path"].read_text()) == config
    if os.name != "nt":
        assert workspace_path.stat().st_mode & 0o777 == 0o700
        assert run["config_path"].stat().st_mode & 0o777 == 0o600
        assert run["log_path"].stat().st_mode & 0o777 == 0o600
    assert not (tmp_path / "input").exists()

    worker.is_running.return_value = False
    gradio_ui.clear_transcription(worker, output_dir, run)


def test_second_start_preserves_active_job(tmp_path):
    upload_path = tmp_path / "recording.mp3"
    upload_path.touch()
    output_dir = tmp_path / "output"
    worker = Mock(pid=1234)
    worker.submit.return_value = 7
    worker.is_running.return_value = True

    first = gradio_ui.start_transcription(
        worker, output_dir, str(upload_path), 0.75, 2.0, 100, 3, 15, "en", "cpu", None
    )
    active_run = first[-1]
    config_contents = active_run["config_path"].read_text()

    second = gradio_ui.start_transcription(
        worker,
        output_dir,
        str(upload_path),
        0.75,
        2.0,
        100,
        3,
        15,
        "en",
        "cpu",
        active_run,
    )

    assert second[0] == "⏳ A transcription is already running."
    assert second[-1] == active_run
    assert worker.submit.call_count == 1
    assert active_run["config_path"].read_text() == config_contents
    assert list(output_dir.glob("job-*")) == [active_run["workspace_path"]]

    gradio_ui.clear_transcription(worker, output_dir, active_run)


def test_same_name_jobs_keep_immutable_results(monkeypatch, tmp_path):
    upload_path = tmp_path / "recording.mp3"
    upload_path.touch()
    output_dir = tmp_path / "output"
    worker = Mock(pid=1234)
    worker.submit.side_effect = [7, 8]

    monkeypatch.setattr(gradio_ui, "preview_transcript", Mock())

    first = gradio_ui.start_transcription(
        worker,
        output_dir,
        str(upload_path),
        0.75,
        2.0,
        100,
        3,
        15,
        "en",
        "cpu",
        None,
    )[-1]
    first["output_path"].write_text("start_time\tend_time\ttext\n", encoding="utf-8")
    worker.result.return_value = (True, None)
    completion = gradio_ui.check_completion(worker, first, None)
    worker.is_running.return_value = False

    second = gradio_ui.start_transcription(
        worker,
        output_dir,
        str(upload_path),
        0.75,
        2.0,
        100,
        3,
        15,
        "en",
        "cpu",
        first,
    )[-1]

    assert completion[-1] == first
    assert first["workspace_path"] != second["workspace_path"]
    assert first["output_path"].read_text(encoding="utf-8") == "start_time\tend_time\ttext\n"
    assert second["output_path"] != first["output_path"]

    worker.is_running.return_value = False
    gradio_ui.clear_transcription(worker, output_dir, second)
    gradio_ui.clear_transcription(worker, output_dir, first)


def test_completion_reports_successful_empty_transcript(monkeypatch, tmp_path):
    output_path = tmp_path / "transcript.csv"
    output_path.write_text("start_time\tend_time\ttext\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    preview = Mock(return_value="empty preview")
    worker = Mock()
    worker.result.return_value = (True, None)
    monkeypatch.setattr(gradio_ui, "preview_transcript", preview)
    run = {
        "job_id": 7,
        "output_path": output_path,
        "config_path": config_path,
        "log_path": tmp_path / "run.log",
        "started": 0,
    }

    result = gradio_ui.check_completion(worker, run, None)

    assert result[0].startswith("✅ No speech was transcribed.")
    assert result[4] == str(output_path)
    preview.assert_called_once_with(True, str(output_path))


@pytest.mark.parametrize(
    ("contents", "expected_error"),
    [
        ("", "could not read transcript"),
        ("garbage\n", "invalid transcript columns"),
    ],
)
def test_completion_reports_invalid_transcript_as_failure(tmp_path, contents, expected_error):
    output_path = tmp_path / "transcript.csv"
    output_path.write_text(contents, encoding="utf-8")
    worker = Mock()
    worker.result.return_value = (True, None)
    run = {
        "job_id": 7,
        "output_path": output_path,
        "config_path": tmp_path / "config.yaml",
        "log_path": tmp_path / "run.log",
        "started": 0,
    }

    result = gradio_ui.check_completion(worker, run, None)

    assert result[0].startswith(f"❌ Transcription failed ({expected_error}")
    assert result[4] is None


def test_clear_cancels_job_and_deletes_artifacts(tmp_path):
    upload_path = tmp_path / "recording.mp3"
    upload_path.touch()
    worker = Mock(pid=1234)
    worker.submit.return_value = 7
    worker.is_running.return_value = True

    output_dir = tmp_path / "output"

    run = gradio_ui.start_transcription(
        worker,
        output_dir,
        str(upload_path),
        0.75,
        2.0,
        100,
        3,
        15,
        "en",
        "cpu",
        None,
    )[-1]
    run["output_path"].write_text("sensitive transcript", encoding="utf-8")

    result = gradio_ui.clear_transcription(worker, output_dir, run)

    worker.terminate.assert_called_once_with()
    assert not run["workspace_path"].exists()
    assert result[-1] is None


def test_show_audio_info_returns_uploaded_file_for_preview(monkeypatch):
    monkeypatch.setattr(
        gradio_ui,
        "mediainfo",
        lambda path: {"duration": "12.34", "sample_rate": "16000", "channels": "1"},
    )

    preview, details = gradio_ui.show_audio_info("recording.mp3")

    assert preview == "recording.mp3"
    assert details == "🕒 Duration: 12.34s\n📊 Sample rate: 16000 Hz\n🔊 Channels: 1"


def test_launch_gradio_disables_network_features(monkeypatch, tmp_path):
    blocks_type = gradio_ui.gr.Blocks
    blocks_init = blocks_type.__init__
    blocks_options = {}
    launch = Mock()
    worker = Mock()

    def capture_blocks_options(self, *args, **kwargs):
        blocks_options.update(kwargs)
        blocks_init(self, *args, **kwargs)

    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "True")
    monkeypatch.setenv("GRADIO_SERVER_NAME", "0.0.0.0")
    monkeypatch.setenv("GRADIO_SHARE", "True")
    monkeypatch.setattr(blocks_type, "__init__", capture_blocks_options)
    monkeypatch.setattr(blocks_type, "launch", launch)
    monkeypatch.setattr(gradio_ui, "detect_device", lambda preferred: "cpu")

    gradio_ui.launch_gradio(tmp_path / "output", worker)

    assert blocks_options["analytics_enabled"] is False
    assert blocks_options["delete_cache"] == (3600, 86400)
    assert launch.call_args.kwargs["share"] is False
    assert launch.call_args.kwargs["server_name"] == "127.0.0.1"
    assert launch.call_args.kwargs["enable_monitoring"] is False
    worker.terminate.assert_called_once_with()
