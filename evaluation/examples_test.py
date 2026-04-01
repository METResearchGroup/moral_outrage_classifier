from evaluation.run_evaluation_harness import EvaluationHarness
import typer

def main(
    input_path: str = typer.Option(..., help="Path to the input CSV file"),
    output_path: str = typer.Option(..., help="Path to the output CSV file"),
    models: list[str] = typer.Option(..., help="List of models to evaluate"),
    max_rows: int = typer.Option(..., help="Maximum number of rows to process"),
    batch_size: int = typer.Option(10, help="Size of each batch for processing")
):
    eh = EvaluationHarness(
        input_path=input_path,
        output_path=output_path,
        batch_size=batch_size,
        models=models,
        max_rows=max_rows
    )

    print("LOADING DATA")
    eh.load_data()
    print("DONE LOADING DATA, NOW RUNNING EVALUATION")
    eh.run_evaluation()
    print("DONE RUNNING EVALUATION, NOW DISPLAYING RESULTS")
    eh.display_results()


if __name__ == "__main__":
    typer.run(main)
