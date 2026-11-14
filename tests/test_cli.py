import os
import sys
import logging
from unittest.mock import Mock

import pytest

from textplease import main


def test_help_exits_successfully(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["textplease", "--help"])

    with pytest.raises(SystemExit) as exit_info:
        main.main()

    assert exit_info.value.code == 0


def test_missing_ffmpeg_exits_without_traceback(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("input_path: input.wav\noutput_path: output.tsv\nmodel_name: test-model\n")
    monkeypatch.setattr(sys, "argv", ["textplease", "--config", str(config_path)])
    monkeypatch.setattr(main.shutil, "which", lambda executable: None)

    with pytest.raises(SystemExit) as exit_info:
        main.main()

    stderr = capsys.readouterr().err
    assert exit_info.value.code == 2
    assert "FFmpeg is required" in stderr
    assert "Traceback" not in stderr


def test_yaml_cannot_change_process_environment(monkeypatch, tmp_path, caplog):
    variable_name = "TEXTPLEASE_PRIVATE_TEST_VALUE"
    private_value = "must-not-enter-environment-or-logs"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "input_path: input.wav\n"
        "output_path: output.tsv\n"
        "model_name: test-model\n"
        f"environment:\n  {variable_name}: {private_value}\n"
    )
    run_pipeline = Mock()
    monkeypatch.delenv(variable_name, raising=False)
    monkeypatch.setattr(sys, "argv", ["textplease", "--config", str(config_path)])
    monkeypatch.setattr(main.shutil, "which", lambda executable: "/usr/bin/ffmpeg")
    monkeypatch.setattr(main, "configure_logging", Mock())
    monkeypatch.setattr(main, "run_transcription_pipeline", run_pipeline)

    main.main()

    assert variable_name not in os.environ
    assert private_value not in caplog.text
    run_pipeline.assert_called_once()


@pytest.mark.parametrize("web", [False, True])
def test_debug_startup_reaches_job_configuration(monkeypatch, tmp_path, web):
    config = tmp_path / "config.yaml"
    config.write_text("input_path: input.wav\noutput_path: output.tsv\nmodel_name: test\n")
    monkeypatch.setattr(sys, "argv", ["textplease", "--debug"] + (["--gradio"] if web else ["--config", str(config)]))
    monkeypatch.setattr(main.shutil, "which", lambda executable: "/usr/bin/ffmpeg")
    logging_setup = Mock()
    launch = Mock()
    runner = Mock()
    monkeypatch.setattr(main, "configure_logging", logging_setup)
    monkeypatch.setattr(main, "launch_gradio", launch)
    monkeypatch.setattr(main, "run_transcription_pipeline", runner)
    main.main()
    logging_setup.assert_called_once_with(level=logging.DEBUG)
    if web:
        launch.assert_called_once_with(debug=True)
    else:
        assert runner.call_args.args[0]["log_level"] == "DEBUG"
