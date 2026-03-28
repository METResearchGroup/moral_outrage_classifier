from models.perspective_api.model import PerspectiveAPIModel
from schemas.responses import MoralOutrage
from lib.testing_utils import print_table, evaluate_model_performance, init_model


def print_perspective_api_table(results: list[MoralOutrage]) -> None:
    title = "Perspective API Results"
    col_headers = ["Text", "Moral Outrage Score"]
    rows = [[res.text, f"{res.moral_outrage_score:.4f}"] for res in results]
    print_table(title, col_headers, rows)

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

    print("\n")

    evaluate_model_performance(
        perspective_model.batch_classify, "This is a test input to evaluate the performance of the Perspective API model."
    )

    print("\n")