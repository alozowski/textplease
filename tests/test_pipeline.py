import csv
from unittest.mock import Mock

from textplease import pipeline, segmenter


def test_similarity_threshold_one_skips_embedding_model(monkeypatch, tmp_path):
    input_path = tmp_path / "input.wav"
    input_path.touch()
    output_path = tmp_path / "output.csv"
    sentence_transformer = Mock(side_effect=AssertionError("Embedding model should not load"))

    monkeypatch.setattr(pipeline, "extract_audio", lambda path: path)
    monkeypatch.setattr(
        pipeline,
        "transcribe_audio",
        lambda *args, **kwargs: [
            {"start_time": "00:00:00.000", "end_time": "00:00:01.000", "text": "Short"},
            {"start_time": "00:00:01.500", "end_time": "00:00:02.000", "text": "fragment"},
        ],
    )
    monkeypatch.setattr(pipeline, "SentenceTransformer", sentence_transformer)
    monkeypatch.setattr(segmenter, "SentenceTransformer", sentence_transformer)
    monkeypatch.setattr(pipeline, "cleanup_temp", lambda original, extracted: False)

    pipeline.run_transcription_pipeline(
        {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "model_name": "test-model",
            "similarity_threshold": 1.0,
        }
    )

    with output_path.open(newline="") as output_file:
        rows = list(csv.DictReader(output_file, delimiter="\t"))

    sentence_transformer.assert_not_called()
    assert [row["text"] for row in rows] == ["Short fragment"]
