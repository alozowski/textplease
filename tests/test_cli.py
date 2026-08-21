import sys

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
