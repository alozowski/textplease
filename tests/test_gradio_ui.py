from unittest.mock import Mock

from textplease import gradio_ui


def test_start_transcription_uses_gradio_cached_file(monkeypatch, tmp_path):
    upload_path = tmp_path / "gradio-cache" / "recording.mp3"
    upload_path.parent.mkdir()
    upload_path.touch()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    worker = Mock(pid=1234)
    worker.submit.return_value = 7

    monkeypatch.setattr(gradio_ui, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(gradio_ui, "PIPELINE_WORKER", worker)

    result = gradio_ui.start_transcription(
        str(upload_path),
        0.75,
        2.0,
        100,
        3,
        15,
        "en",
        "cpu",
    )

    config, log_path = worker.submit.call_args.args
    assert config["input_path"] == str(upload_path)
    assert log_path == str(output_dir / "recording_run.log")
    assert result[-1]["output_path"] == output_dir / "recording_transcript.csv"
    assert not (tmp_path / "input").exists()


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
    monkeypatch.setattr(gradio_ui, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(gradio_ui, "PIPELINE_WORKER", worker)
    monkeypatch.setattr(gradio_ui, "detect_device", lambda preferred: "cpu")

    gradio_ui.launch_gradio()

    assert blocks_options["analytics_enabled"] is False
    assert blocks_options["delete_cache"] == (3600, 86400)
    assert launch.call_args.kwargs["share"] is False
    assert launch.call_args.kwargs["server_name"] == "127.0.0.1"
    assert launch.call_args.kwargs["enable_monitoring"] is False
    worker.shutdown.assert_called_once_with()
