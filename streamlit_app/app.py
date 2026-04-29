import json
import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from streamlit_app.constants import (
    MODELS,
    RUN_STATE_COMPLETED,
    RUN_STATE_COMPLETED_WITH_FAILURES,
    RUN_STATE_FAILED,
    RUN_STATE_READY,
    RUN_STATE_RUNNING,
)
from evaluation.column_name_conversion import COLUMN_NAME_CONVERSION
from streamlit_app.harness_runner import _run_harness
from streamlit_app.utils import (
    count_csv_rows,
    ensure_dirs,
    has_gold_label,
    read_csv_columns,
    save_uploaded_file,
    validate_columns,
)

load_dotenv(Path(__file__).parent.parent / ".env")


def _init_session_state():
    ensure_dirs()
    defaults = {
        "run_state": RUN_STATE_READY,
        "run_thread": None,
        "progress": {"current": 0, "total": 0},
        "_result_state": None,
        "result": None,
        "run_error": None,
        "dataset_path": None,
        "dataset_has_gold": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _start_run(dataset_path: Path, model: str):
    state_lock = threading.Lock()

    progress_state = {"current": 0, "total": 0}
    result_state = {"done": False, "error": None, "output_path": None}

    with state_lock:
        st.session_state.state_lock = state_lock
        st.session_state.progress = progress_state
        st.session_state._result_state = result_state
        st.session_state.run_state = RUN_STATE_RUNNING
        st.session_state.result = None
        st.session_state.run_error = None

    thread = threading.Thread(
        target=_run_harness,
        args=(dataset_path, model, progress_state, result_state, state_lock),
        daemon=True,
    )
    st.session_state.run_thread = thread
    thread.start()


def _render_progress_bar() -> None:
    with st.session_state.state_lock:
        current = st.session_state.progress["current"]
        total = st.session_state.progress["total"]

    if total > 0:
        st.progress(current / total, text=f"{current:,} / {total:,} rows")
    else:
        st.progress(0.0, text="Starting…")
    time.sleep(0.5)
    st.rerun()


def _finalize_run(result_state: dict) -> None:
    if result_state.get("error"):
        st.session_state.run_state = RUN_STATE_FAILED
        st.session_state.run_error = result_state["error"]
    else:
        output_path: Path = result_state["output_path"]
        deadletter_path = output_path / "deadletter.csv"
        failed_count = count_csv_rows(deadletter_path) if deadletter_path.exists() else 0
        st.session_state.result = {
            "output_path": output_path,
            "failed_count": failed_count,
        }
        st.session_state.run_state = (
            RUN_STATE_COMPLETED_WITH_FAILURES if failed_count > 0 else RUN_STATE_COMPLETED
        )
    st.rerun()


def _poll_running_state():
    thread: threading.Thread = st.session_state.run_thread

    if thread and thread.is_alive():
        _render_progress_bar()
    else:
        with st.session_state.state_lock:
            result_state: dict = st.session_state._result_state
        _finalize_run(result_state)


def _unpack_run_result() -> tuple[Path, Path, int]:
    result = st.session_state.result
    output_path: Path = result["output_path"]
    failed_count: int = result["failed_count"]
    output_csv = output_path / "output.csv"
    metrics_path = output_path / "metrics.json"
    return output_csv, metrics_path, failed_count


def _render_completion_status(failed_count: int) -> None:
    if st.session_state.run_state == RUN_STATE_COMPLETED:
        st.success("Labeling complete.")
    else:
        st.warning(
            f"Labeling complete with **{failed_count}** failed row(s). "
            "Export the output below, then rerun the dataset to recover failed rows."
        )


def _render_evaluation_metrics(metrics_path: Path) -> None:
    if not (metrics_path.exists() and st.session_state.dataset_has_gold):
        return
    st.subheader("Evaluation metrics")
    with open(metrics_path) as f:
        metrics = json.load(f)
    for model_name, m in metrics.items():
        st.write(f"**{model_name}**")
        cols = st.columns(4)
        cols[0].metric("Accuracy", f"{m['accuracy']:.4f}")
        cols[1].metric("Precision", f"{m['precision']:.4f}")
        cols[2].metric("Recall", f"{m['recall']:.4f}")
        cols[3].metric("F1", f"{m['f1_score']:.4f}")
        st.caption(f"Evaluated on {m['total_samples']} samples")


def _render_export_button(output_csv: Path) -> None:
    with open(output_csv, "rb") as f:
        st.download_button(
            label="Export output CSV",
            data=f,
            file_name="labeled_output.csv",
            mime="text/csv",
        )


def _render_output_preview(output_csv: Path) -> None:
    st.subheader("Output preview")
    df = pd.read_csv(output_csv)
    st.dataframe(df.head(50), width="stretch")


def _render_results():
    output_csv, metrics_path, failed_count = _unpack_run_result()
    _render_completion_status(failed_count)
    _render_output_preview(output_csv)
    _render_evaluation_metrics(metrics_path)
    _render_export_button(output_csv)


def _configure_page() -> None:
    st.set_page_config(page_title="Moral Outrage Labeler", layout="centered")
    st.title("Moral Outrage Labeler")


def _get_run_state() -> tuple[str, bool]:
    run_state = st.session_state.run_state
    is_running = run_state == RUN_STATE_RUNNING
    return run_state, is_running


def _reject_upload(save_path: Path, missing: list[str]) -> None:
    save_path.unlink(missing_ok=True)
    aliases = {c: COLUMN_NAME_CONVERSION[c] for c in missing}
    st.error(
        f"Upload rejected — required field(s) not found: "
        + ", ".join(f"`{c}` (accepted names: {aliases[c]})" for c in missing)
    )
    st.session_state.dataset_path = None


def _accept_upload(save_path: Path, columns: list[str]) -> None:
    row_count = count_csv_rows(save_path)
    has_gold = has_gold_label(columns)
    st.session_state.dataset_path = save_path
    st.session_state.dataset_has_gold = has_gold
    st.caption(
        f"{row_count} rows · columns: {', '.join(columns)}"
        + (" · gold_label present — evaluation metrics will be shown" if has_gold else "")
    )


def _render_upload_section(is_running: bool) -> None:
    st.header("1. Upload dataset")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], disabled=is_running)
    if uploaded_file is not None:
        save_path = save_uploaded_file(uploaded_file)
        columns = read_csv_columns(save_path)
        missing = validate_columns(columns)
        if missing:
            _reject_upload(save_path, missing)
        else:
            _accept_upload(save_path, columns)


def _render_model_selection_section(is_running: bool) -> str:
    st.header("2. Select model")
    return st.selectbox("Model", MODELS, disabled=is_running)


def _render_run_section(is_running: bool, dataset_path: Path | None, selected_model: str) -> None:
    st.header("3. Run")
    if st.button("Start labeling", disabled=is_running or dataset_path is None):
        _start_run(dataset_path, selected_model)
        st.rerun()
    if dataset_path is None and not is_running:
        st.caption("Upload a dataset above to enable the run.")


def _render_progress_section() -> None:
    st.header("Running")
    _poll_running_state()


def _render_results_section(run_state: str) -> None:
    if run_state in (RUN_STATE_COMPLETED, RUN_STATE_COMPLETED_WITH_FAILURES):
        st.header("Results")
        _render_results()
    elif run_state == RUN_STATE_FAILED:
        st.error(f"Run failed: {st.session_state.run_error}")
        if st.button("Reset"):
            st.session_state.run_state = RUN_STATE_READY
            st.rerun()


def main():
    _configure_page()
    _init_session_state()
    run_state, is_running = _get_run_state()

    _render_upload_section(is_running)
    selected_model = _render_model_selection_section(is_running)
    _render_run_section(is_running, st.session_state.dataset_path, selected_model)

    if is_running:
        _render_progress_section()
        return

    _render_results_section(run_state)


if __name__ == "__main__":
    main()
