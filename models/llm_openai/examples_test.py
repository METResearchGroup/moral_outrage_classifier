from models.llm_openai.model import OpenAIModel
from schemas.responses import MoralOutrage
from lib.testing_utils import print_table, evaluate_model_performance, init_model


def print_openai_table(results: list[MoralOutrage]) -> None:
    title = "OpenAI Inference Results"
    col_headers = ["Text", "Moral Outrage Score"]
    rows = [[res.text, f"{res.moral_outrage_score:.4f}"] for res in results]
    print_table(title, col_headers, rows)

def verify_diff_cases(model: OpenAIModel) -> None:
    test_cases = [
        "I love this product! It's amazing.",
        "The movie was okay, not great but not terrible either.",
        """I'm furious and disgusted that the company is exploiting child labor
        and the executives responsible should be held 
        accountable and punished for their actions.
        """,
        """It makes me sick and angry that people are dumping toxic waste into the river,
        and those responsible deserve to be blamed and face serious consequences.
        """,
        "What a wonderful day! I'm so happy.",
    ]

    results = model.batch_classify(test_cases)
    print_openai_table(results)

if __name__ == "__main__":
    print("\n")
    openai_model = init_model(OpenAIModel)

    verify_diff_cases(openai_model)

    print("\n")

    evaluate_model_performance(
        openai_model.batch_classify, "This is a test input to evaluate the performance of the Perspective API model."
    )

    print("\n")
    