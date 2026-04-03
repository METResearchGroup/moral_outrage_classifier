import collections
import csv
from models.perspective_api.model import PerspectiveAPIModel
from evaluation.dataloader import DataLoader
from pathlib import Path
from lib.testing_utils import print_table
from schemas.responses import MoralOutrage


MODEL_REGISTRY: dict[str, type] = {
    "perspective_api": PerspectiveAPIModel,
}

VALID_MODELS = list(MODEL_REGISTRY.keys())

class EvaluationHarness:
    def __init__(
          self,
          input_path: str,
          output_path: str,
          batch_size: int,
          models: list[str],
          max_rows: int | float = float("inf"),
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.models = models
        self.dataloaders: dict[str, DataLoader] = {
            model_name: DataLoader(input_path=input_path, output_path=output_path, batch_size=batch_size, model_name=model_name, max_rows=max_rows)
            for model_name in models
        }

    @classmethod
    def validate_models(cls, models: list[str]):
        if len(models) == 0 or any(model not in VALID_MODELS for model in models):
            raise ValueError(f"Please provide a valid model. {models} is invalid")

    def load_data(self) -> None:
        for dataloader in self.dataloaders.values():
            dataloader.load_data()

    def _get_model_output_path(self, model_name: str) -> str:
        path = Path(self.output_path)
        return str(path.parent / f"{path.stem}_{model_name}{path.suffix}")

    def _write_to_model_csv(self, path: str, model_name: str, batch: list[dict[str, str | int]], predictions: list[MoralOutrage]) -> None:
        with open(path, "a") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "dataset", "text", "gold_label", "pred_label", "is_correct"])
            if f.tell() == 0:
                writer.writeheader()
            for sample, prediction in zip(batch, predictions, strict=True):
                if prediction is None:
                    pred_label = None
                else:
                    if model_name == "perspective_api":
                        pred_label = 1 if prediction.moral_outrage_score > 0.7 else 0
                    else:
                        pred_label = prediction.moral_outrage_score 

                is_correct = int(sample["gold_label"]) == int(pred_label) if pred_label is not None and sample["gold_label"] is not None else None

                writer.writerow({
                    "id": sample["id"],
                    "dataset": self.input_path,
                    "text": sample["text"],
                    "gold_label": sample["gold_label"],
                    "pred_label": pred_label,
                    "is_correct": is_correct,
                })

    def _run_model_evaluation(self, model_name: str) -> None:
        model = MODEL_REGISTRY[model_name]()
        path = self._get_model_output_path(model_name)
        for batch in self.dataloaders[model_name]:
            texts = [sample["text"] for sample in batch]

            try:
                predictions = model.batch_classify(texts)
                self._write_to_model_csv(path, model_name, batch, predictions)
            except Exception as e:
                print(f"Error during model evaluation: {e}")
                
    def _copy_model_results_to_merged_csv(self, path: str, writer: csv.DictWriter, model_name: str) -> None:
        with open(path, "r") as f_in:
            reader = csv.DictReader(f_in)
            for row in reader:
                row["model"] = model_name
                writer.writerow(row)

    def _merge_model_results(self) -> None:
        with open(self.output_path, "a") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=["id", "dataset", "text", "gold_label", "pred_label", "is_correct", "model"])
            if f_out.tell() == 0:
                writer.writeheader()
            for model_name in self.models:
                path = self._get_model_output_path(model_name)
                self._copy_model_results_to_merged_csv(path, writer, model_name)

    def _delete_temp_model_csv(self, model_name: str) -> None:
        path = Path(self._get_model_output_path(model_name))
        path.unlink(missing_ok=True)

    def run_evaluation(self) -> None:
        for model in self.models:
            self._run_model_evaluation(model)

        self._merge_model_results()

        for model in self.models:
            self._delete_temp_model_csv(model)

    def _get_rows_by_model_dict(self) -> dict[str, list[dict[str, str | int]]]:
        rows_by_model = collections.defaultdict(list)
        with open(self.output_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_name = row["model"]
                rows_by_model[model_name].append(row)
        
        return rows_by_model

    def _calculate_run_metrics(self, rows: list[dict[str, str | int]]) -> tuple[int, float, float, float, float]:
        # csv.DictWfriter writes "" when it wants to write None to a csv file, so need to check for "" when a prediction failed from HttpError
        total_samples = sum(1 for row in rows if row["pred_label"] != "" and row["gold_label"] != "")
            
        tp = sum(1 for row in rows
                 if row["pred_label"] != ""
                 and row["gold_label"] != ""
                 and int(row["pred_label"]) == 1
                 and int(row["gold_label"]) == 1)
        tn = sum(1 for row in rows
                 if row["pred_label"] != ""
                 and row["gold_label"] != ""
                 and int(row["pred_label"]) == 0
                 and int(row["gold_label"]) == 0)
        fp = sum(1 for row in rows
                 if row["pred_label"] != ""
                 and row["gold_label"] != ""
                 and int(row["pred_label"]) == 1
                 and int(row["gold_label"]) == 0)
        fn = sum(1 for row in rows
                 if row["pred_label"] != ""
                 and row["gold_label"] != ""
                 and int(row["pred_label"]) == 0
                 and int(row["gold_label"]) == 1)

        accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return total_samples, accuracy, precision, recall, f1

    def _prepare_table_rows(self) -> list[list[str]]:
        rows_by_model = self._get_rows_by_model_dict()
        
        table_rows = []
        for model_name, rows in rows_by_model.items():
            total_samples, accuracy, precision, recall, f1 = self._calculate_run_metrics(rows)
            table_rows.append([model_name, f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", str(total_samples)])

        return table_rows

    def display_results(self) -> None:
        table_rows = self._prepare_table_rows()
        print_table(
            "Evaluation Results",
            ["Model", "Accuracy", "Precision", "Recall", "F1", "Total Samples"],
            table_rows
        )


