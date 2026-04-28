from pathlib import Path

from evaluation.model_registry import MODEL_REGISTRY

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOADS_DIR = PROJECT_ROOT / "streamlit_app" / "uploads"
OUTPUTS_DIR = PROJECT_ROOT / "streamlit_app" / "outputs"
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

MODELS = list(MODEL_REGISTRY.keys())

RUN_STATE_READY = "ready"
RUN_STATE_RUNNING = "running"
RUN_STATE_COMPLETED = "completed"
RUN_STATE_COMPLETED_WITH_FAILURES = "completed_with_failures"
RUN_STATE_FAILED = "failed"
