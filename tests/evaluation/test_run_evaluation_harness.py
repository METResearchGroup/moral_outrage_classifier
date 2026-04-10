import csv
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from evaluation.run_evaluation_harness import EvaluationHarness, RETRIES

FAKE_BATCH: list[dict[str, str]] = [
    {"id": "1", "text": "hello world", "gold_label": "1"},
    {"id": "2", "text": "this is outrageous", "gold_label": "0"},
]


@pytest.fixture
def harness(tmp_path: Path, mock_model: MagicMock) -> EvaluationHarness:
    with patch("evaluation.run_evaluation_harness.DataLoader"), \
         patch.dict("evaluation.run_evaluation_harness.MODEL_REGISTRY", {"perspective_api": MagicMock(return_value=mock_model)}):
        h = EvaluationHarness(
            input_path="unused",
            output_path=str(tmp_path / "output"),
            batch_size=2,
            models=["perspective_api"],
            timestamp="test_run",
        )

    h.dataloaders["perspective_api"] = [FAKE_BATCH]  # type: ignore[assignment]
    return h


@pytest.fixture
def mock_model() -> MagicMock:
    mm = MagicMock()
    mm.batch_classify.return_value = [
        MagicMock(moral_outrage_score=0.8),
        MagicMock(moral_outrage_score=0.2),
    ]
    return mm


def read_deadletter(harness: EvaluationHarness) -> list[dict[str, str]]:
    with open(harness.new_output_path / "deadletter.csv") as f:
        return list(csv.DictReader(f))


class TestDeadletter:
    def test_deadletter_written_on_batch_failure(self, harness: EvaluationHarness) -> None:
        with patch.object(harness, "_process_batch", side_effect=Exception("API error")):
            harness._run_model_evaluation("perspective_api")

        assert (harness.new_output_path / "deadletter.csv").exists()

    def test_deadletter_contains_correct_fields(self, harness: EvaluationHarness) -> None:
        with patch.object(harness, "_process_batch", side_effect=Exception("API error")):
            harness._run_model_evaluation("perspective_api")

        rows = read_deadletter(harness)
        assert rows[0] == {"id": "1", "text": "hello world", "model": "perspective_api"}
        assert rows[1] == {"id": "2", "text": "this is outrageous", "model": "perspective_api"}

    def test_deadletter_appends_across_multiple_failed_batches(self, harness: EvaluationHarness) -> None:
        batch1: list[dict[str, str]] = [{"id": "1", "text": "a", "gold_label": "1"}]
        batch2: list[dict[str, str]] = [{"id": "2", "text": "b", "gold_label": "0"}]
        harness.dataloaders["perspective_api"] = [batch1, batch2]  

        with patch.object(harness, "_process_batch", side_effect=Exception("fail")):
            harness._run_model_evaluation("perspective_api")

        rows = read_deadletter(harness)
        assert len(rows) == 2 
        assert rows[0]["id"] == "1"
        assert rows[1]["id"] == "2"

    def test_no_deadletter_on_success(self, harness: EvaluationHarness, mock_model: MagicMock) -> None:
        with patch.dict("evaluation.run_evaluation_harness.MODEL_REGISTRY", {"perspective_api": MagicMock(return_value=mock_model)}):
            harness._run_model_evaluation("perspective_api")

        assert not (harness.new_output_path / "deadletter.csv").exists()


class TestRetries:
    def test_retries_exhausted_before_deadletter(self, harness: EvaluationHarness, mock_model: MagicMock) -> None:
        mock_model.batch_classify.side_effect = Exception("transient error")

        with patch.dict("evaluation.run_evaluation_harness.MODEL_REGISTRY", {"perspective_api": MagicMock(return_value=mock_model)}), \
            patch("time.sleep"):  # suppress tenacity wait between retries
            harness._run_model_evaluation("perspective_api")

        assert mock_model.batch_classify.call_count == RETRIES
        assert (harness.new_output_path / "deadletter.csv").exists()

    def test_no_deadletter_if_retry_succeeds(self, harness: EvaluationHarness, mock_model: MagicMock) -> None:
        # Fails on first attempt, succeeds on retry
        mock_model.batch_classify.side_effect = [
            Exception("transient"),
            mock_model.batch_classify.return_value,
        ]

        with patch.dict("evaluation.run_evaluation_harness.MODEL_REGISTRY", {"perspective_api": MagicMock(return_value=mock_model)}), \
            patch("time.sleep"):
            harness._run_model_evaluation("perspective_api")

        assert mock_model.batch_classify.call_count == 2
        assert not (harness.new_output_path / "deadletter.csv").exists()
