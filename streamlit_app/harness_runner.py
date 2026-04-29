from pathlib import Path

from evaluation.run_evaluation_harness import EvaluationHarness
from lib.timestamp_utils import get_current_timestamp
from streamlit_app.constants import PROJECT_ROOT, OUTPUTS_DIR


def _run_harness(dataset_path: Path, model: str, progress_state: dict, result_state: dict, state_lock):
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
        with state_lock:
            progress_state["current"] = current
            progress_state["total"] = total

    try:
        harness.run_evaluation(progress_callback=callback)
        with state_lock:
            result_state["output_path"] = OUTPUTS_DIR / timestamp
    except Exception as e:
        with state_lock:
            result_state["error"] = str(e)
    finally:
        with state_lock:
            result_state["done"] = True
