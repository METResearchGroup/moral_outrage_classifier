import csv
import math
from pathlib import Path
from typing import Iterator

column_name_conversion = {
    "id": ["id", "tweet_id"],
    "text": ["text", "body"],
    "gold_label": ["gold_label", "outrage", "pers_outrage_label"],
}


class DataLoader:
    def __init__(self, input_path: str, output_path: str, batch_size: int, model_name: str, max_rows: int | float = float('inf')):
        self.data: list[dict[str, str | int]] = [] # stores the records in RAM after filtering out already processed records

        path = Path(input_path)
        if not path.is_file():
            raise FileNotFoundError(f"Input path {input_path} is not a valid file.")
        self.input_path = input_path
        self.output_path = output_path
        self.batch_size = batch_size
        self.max_rows = max_rows

        model_output_path = Path(output_path)
        # ex: evaluation/output.csv -> evaluation/output_perspective_api.csv
        self.model_output_path = str(model_output_path.parent / f"{model_output_path.stem}_{model_name}{model_output_path.suffix}")

    # puts all of the id's from output path into a set
    def _return_already_processed_ids(self) -> set[str]:
        already_processed_ids = set()
        for path in [self.output_path, self.model_output_path]:
            try:
                with open(path, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        already_processed_ids.add(row["id"])
            except FileNotFoundError:
                pass
        return already_processed_ids
    
    def _get_field_value(self, field_name: str, row: dict):
        """Get a value for a specific field in the .csv file.
           The field can be mapped to the canonical field names using the column_name_conversion dictionary.
        """ 
        value = next((row[key] for key in column_name_conversion[field_name] if key in row), None)  
        return value 
    
    def _post_already_processed(self, row: dict[str, str], already_processed_ids: set[str]) -> bool:
        post_id = self._get_field_value("id", row)
        return post_id in already_processed_ids

    def _get_new_row_data(self, row: dict[str, str]) -> dict[str, str | int]:
        """
        Appends new data to the list of new_data if the post id is not in the set of already processed ids.
        Assumes that the row is an unprocessed record. 
        
        Args:
            row (dict[str, str]): A dictionary representing a row from the input CSV file.
                                  Contains keys that can be mapped to "id", "text", and "gold_label" using the column_name_conversion dictionary.
            already_processed_ids (set[str]): A set of post ids that have already been processed.
            new_data (list[dict[str, str | int]]): A list to which new

        Returns:
            None: This function does not return anything, it modifies the new_data list in place.
        """
        post_id = self._get_field_value("id", row)
        text = self._get_field_value("text", row) 
        gold_label_str = self._get_field_value("gold_label", row)

        try:
            gold_label = int(gold_label_str) if gold_label_str is not None else None
        except (ValueError, TypeError):
            gold_label = None

        return {"text": text, "gold_label": gold_label, "id": post_id}

    def _return_new_records(self, already_processed_ids: set[str]) -> list[dict[str, str | int]]:
        """
        Uses the set of already processed id's to filter out records from input path
        
        Args:
            already_processed_ids: A set of post ids that have already been processed.

        Returns:
            A list of dictionaries representing new records that have not been processed yet.
            Each dictionary contains the keys "text", "gold_label", and "id".
        """
        new_data = []
        with open(self.input_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if self._post_already_processed(row, already_processed_ids):
                    continue

                new_data.append(self._get_new_row_data(row))
                if len(new_data) >= self.max_rows:
                    break

        return new_data

    def load_data(self):
        new_data = self.filter_already_processed_records()
        self.data.extend(new_data)

    def filter_already_processed_records(self) -> list[dict[str, str | int]]:
        already_processed_ids = self._return_already_processed_ids()
        new_data = self._return_new_records(already_processed_ids)

        return new_data

    def __iter__(self) -> Iterator[list[dict[str, str | int]]]:
        for i in range(0, len(self.data), self.batch_size):
            yield self.data[i:i + self.batch_size]

    def __len__(self) -> int:
        return math.ceil(len(self.data) / self.batch_size)