from rich.console import Console
from rich.table import Table
from collections.abc import Callable
import time
from typing import TypeVar, ParamSpec
from functools import wraps

from models.perspective_api.model import PerspectiveAPIModel
from schemas.responses import MoralOutrage

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")


def print_table(title: str, col_headers: list[str], rows: list[list[str]]) -> None:
    console=Console()
    table = Table(title=title)
    for header in col_headers:
        table.add_column(header)
    for row in rows:
        table.add_row(*row)
    console.print(table)

def print_perspective_api_table(results: list[MoralOutrage]) -> None:
    title = "Perspective API Results"
    col_headers = ["Text", "Moral Outrage Score"]
    rows = [[res.text, f"{res.moral_outrage_score:.4f}"] for res in results]
    print_table(title, col_headers, rows)


def store_elapsed_times(inference_func: Callable, counts: list[int], prompt: str) -> list[float]:
    elapsed_times = []
    for count in counts:
        inputs = [prompt] * count
        start = time.perf_counter()
        inference_func(inputs)
        elapsed = time.perf_counter() - start
        elapsed_times.append(elapsed)

    return elapsed_times

def evaluate_model_performance(inference_func: Callable , prompt: str) -> None:
    counts = [1, 10, 100, 1000, 10000]

    # store the elapsed time for each count of inputs
    elapsed_times = store_elapsed_times(inference_func, counts, prompt)

    # calculate the average time per input for each count. this will be used as a column in the table
    time_per_input = [(count / elapsed) for (count, elapsed) in zip(counts, elapsed_times, strict=True)]
    
    # prepare the rows for the table
    # each row: [input count, elapsed time, average inference time]
    rows = []
    for count, elapsed, time_per in zip(counts, elapsed_times, time_per_input, strict=True):
        rows.append([str(count), f"{elapsed:.4f}", f"{time_per:.4f}"])

    print_table(
        "Model Performance",
        ["Input Count", "Elapsed Time (s)", "Avg inference time (iters/s)"],
        rows
    )

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



def verify_diff_cases(model: PerspectiveAPIModel) -> None:
    test_cases = [
        "I love this product! It's amazing.",
        "This is the worst experience I've ever had.",
        "The movie was okay, not great but not terrible either.",
        "You are a terrible person and I hate you.",
        "What a wonderful day! I'm so happy."
    ]

    results = model.batch_classify(test_cases)
    print_perspective_api_table(results)

if __name__ == "__main__":
    print("\n")
    perspective_model = init_model(PerspectiveAPIModel)

    verify_diff_cases(perspective_model)

    evaluate_model_performance(
        perspective_model.batch_classify, "This is a test input to evaluate the performance of the Perspective API model."
    )

    print("\n")