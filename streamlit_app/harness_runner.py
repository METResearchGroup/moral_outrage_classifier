import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.run_evaluation_harness import EvaluationHarness
from lib.timestamp_utils import get_current_timestamp
from constants import PROJECT_ROOT, OUTPUTS_DIR


def _run_harness(dataset_path: Path, model: str, progress_state: dict, result_state: dict):
    timestamp = get_current_timestamp()
    harness = EvaluationHarness(
        input_path=str(dataset_path.relative_to(PROJECT_ROOT)),
        output_path=str(OUTPUTS_DIR.relative_to(PROJECT_ROOT)),
        batch_size=10,
        models=[model],
        timestamp=timestamp,
    )
    harness.load_data()

    def callback(current: int, total: int):
        progress_state["current"] = current
        progress_state["total"] = total

    try:
        harness.run_evaluation(progress_callback=callback)
        result_state["output_path"] = OUTPUTS_DIR / timestamp
        result_state["done"] = True
    except Exception as e:
        result_state["error"] = str(e)
        result_state["done"] = True
