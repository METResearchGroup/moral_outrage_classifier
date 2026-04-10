import collections
import csv
import json

from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

from models.base import BaseModel
from models.perspective_api.model import PerspectiveAPIModel, PROB_LABEL_THRESHOLD
from evaluation.dataloader import DataLoader
from pathlib import Path
from lib.testing_utils import print_table
from schemas.responses import MoralOutrage

FIELDNAMES = ["id", "dataset", "text", "gold_label", "pred_label", "is_correct", "model"]

MODEL_REGISTRY: dict[str, type] = {
    "perspective_api": PerspectiveAPIModel,
}

RETRIES = 3

VALID_MODELS = list(MODEL_REGISTRY.keys())

RETRY_WAIT_TIME = 2

class EvaluationHarness:
    def __init__(
          self,
          input_path: str,
          output_path: str,
          batch_size: int,
          models: list[str],
          timestamp: str,
          max_rows: int | float = float("inf"),
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.new_output_path = self.output_path / timestamp
        self.new_output_path.mkdir(parents=True, exist_ok=True)
        self.new_output_file = self.new_output_path / "output.csv"
        self.timestamp = timestamp
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

    def _get_model_output_path(self, output_path: Path, model_name: str) -> Path:
        return output_path / f"{model_name}.csv"
    
    def _get_deadletter_path(self, output_path: Path) -> Path:
        return output_path / "deadletter.csv"

    def _write_to_model_csv(self, path: Path, model_name: str, batch: list[dict[str, str | int]], predictions: list[MoralOutrage]) -> None:
        with open(path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            for sample, prediction in zip(batch, predictions, strict=True):
                if prediction is None:
                    pred_label = None
                else:
                    if model_name == "perspective_api":
                        pred_label = 1 if prediction.moral_outrage_score > PROB_LABEL_THRESHOLD else 0
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
                    "model": model_name,
                })
    
    @retry(stop=stop_after_attempt(RETRIES), wait=wait_fixed(RETRY_WAIT_TIME))
    def _process_batch(
        self, 
        texts: list[str], 
        path: Path, 
        model: BaseModel, 
        model_name: str, 
        batch: list[dict[str, str | int]],
    ) -> None:
        predictions = model.batch_classify(texts)
        self._write_to_model_csv(path, model_name, batch, predictions)

    def _write_to_deadletter_csv(self, path, model_name, batch):
        deadletter_file = self._get_deadletter_path(path)

        with open(deadletter_file, "a") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "text", "model"])
            if f.tell() == 0:
                writer.writeheader()
            
            for sample in batch:
                writer.writerow({
                    "id": sample.get("id", "NO ID FOUND"),
                    "text": sample.get("text", "NO TEXT FOUND"),
                    "model": model_name
                })

    def _run_model_evaluation(self, model_name: str) -> None:
        model = MODEL_REGISTRY[model_name]()
        path = self._get_model_output_path(self.new_output_path, model_name)
        for batch in tqdm(self.dataloaders[model_name], desc=f"Evaluating {model_name}"):
            texts = [sample["text"] for sample in batch]

            try:
                self._process_batch(texts, path, model, model_name, batch)

            except Exception as e:
                self._write_to_deadletter_csv(self.new_output_path, model_name, batch)
                print(f"Error during model evaluation: {e}")
                
    def _copy_model_results_to_merged_csv(self, path: str, writer: csv.DictWriter) -> None:
        """
        Reads from a specific csv file and copies all of its rows into a final merged csv file.

        Args:
            path (str): The file path to the model-specific csv file.
            writer (csv.DictWriter): A csv.DictWriter object that is already set up to write to the final merged csv file.
        
        Returns:
            None: This function does not return anything, it writes rows to the final merged csv file using the provided writer object.
        """
        if not Path(path).exists():
            return
        with open(path, "r") as f_in:
            reader = csv.DictReader(f_in)
            for row in reader:
                writer.writerow(row)

    def _merge_model_results(self) -> None:
        """
        Reads from each model-specific csv file and copies all of the rows into a final merged csv file. 
        
        We are writing and reading into model-specific csv files in order to avoid RAM issues 
        that would arise from storing all predictions in RAM before writing to a final csv file.
        
        Our algorithm goes through each model sequentially, so it's best to read and write results to individual model csv files.
        
        Args:
            None: This function does not take in any arguments, it uses the instance variable self.models to determine the file names.

        Returns:
            None: This function does not return anything, it writes rows to the final merged csv file.
        """
        with open(self.new_output_file, "w") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=FIELDNAMES)
            writer.writeheader()
            for model_name in self.models:
                path = self._get_model_output_path(self.new_output_path, model_name)
                self._copy_model_results_to_merged_csv(path, writer)

    def _delete_temp_model_csv(self, model_name: str) -> None:
        path = Path(self._get_model_output_path(self.new_output_path, model_name))
        path.unlink(missing_ok=True)

    def _create_metrics_json(self, rows_by_model: dict[str, list[dict[str, str | int]]]) -> dict:
        metrics_data = {}

        for model_name, rows in rows_by_model.items():
            total_samples, accuracy, precision, recall, f1 = self._calculate_run_metrics(rows)
            metrics_data[model_name] = {
                "total_samples": total_samples,
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4)
            }
        return metrics_data

    def _write_metrics_file(self):
        rows_by_model = self._get_rows_by_model_dict()
        
        metrics_json = self._create_metrics_json(rows_by_model)

        # write to file
        metrics_file = self.new_output_path / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics_json, f, indent=2)

    def run_evaluation(self) -> None:
        for model in self.models:
            self._run_model_evaluation(model)

        self._merge_model_results()

        for model in self.models:
            self._delete_temp_model_csv(model)

        self._write_metrics_file()

    def _get_rows_by_model_dict(self) -> dict[str, list[dict[str, str | int]]]:
        rows_by_model = collections.defaultdict(list)
        with open(self.new_output_file, "r") as f:
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
