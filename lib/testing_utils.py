import time
from typing import Callable, TypeVar, ParamSpec
from functools import wraps
try:
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")

def track_runtime(should_print: bool) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                elapsed: float = time.perf_counter() - start
                if should_print:
                    print(f"[{function.__name__}] ({elapsed:.4f}s)\n") 

        return wrapper

    return decorator

@track_runtime(should_print=True)
def init_model(model_class: type[T]) -> T:
    return model_class()

def print_table(title: str, col_headers: list[str], rows: list[list[str]]) -> None:
    if not RICH_AVAILABLE:
        print("Rich library not available. Install it with 'uv sync --extra rich' to see formatted tables.")
        return
    console=Console()
    table = Table(title=title)
    for header in col_headers:
        table.add_column(header)
    for row in rows:
        table.add_row(*row)
    console.print(table)

def time_inference_execution(inference_func: Callable, inputs: list[str]) -> float:
    start = time.perf_counter()
    inference_func(inputs)
    return time.perf_counter() - start

def _collect_execution_times_by_count(inference_func: Callable, counts: list[int], prompt: str) -> list[float]:
    elapsed_times = []
    for count in counts:
        elapsed = time_inference_execution(inference_func, [prompt] * count)
        elapsed_times.append(elapsed)

    return elapsed_times

def _prepare_table_rows(counts: list[int], execution_times: list[float], time_per_input: list[float]) -> list[list[str]]:
    rows = []
    for count, elapsed, time_per in zip(counts, execution_times, time_per_input, strict=True):
        rows.append([str(count), f"{elapsed:.4f}", f"{time_per:.4f}"])
    return rows

def _display_results_in_table(counts: list[int], execution_times: list[float]) -> None:
    # calculate the average time per input for each count. this will be used as a column in the table
    time_per_input = [(count / elapsed) if elapsed > 0 else float('inf') 
                      for (count, elapsed) in zip(counts, execution_times, strict=True)]
    
    rows = _prepare_table_rows(counts, execution_times, time_per_input)

    print_table(
        "Model Performance",
        ["Input Count", "Elapsed Time (s)", "Avg inference time (iters/s)"],
        rows
    )

def evaluate_model_performance(inference_func: Callable , prompt: str) -> None:
    counts = [1, 10, 100, 1000, 10000]

    # store the elapsed time for each count of inputs
    execution_times = _collect_execution_times_by_count(inference_func, counts, prompt)

    _display_results_in_table(counts, execution_times)

