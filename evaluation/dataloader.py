import csv
import math
import json
from pathlib import Path
from typing import Iterator

column_name_conversion = {
    "id": ["id", "tweet_id"],
    "text": ["text", "body"],
    "gold_label": ["gold_label", "outrage", "pers_outrage_label"],
}


class DataLoader:
    def __init__(self, input_path: Path, output_path: Path, batch_size: int, model_name: str, max_rows: int | float = float('inf')):
        self.data: list[dict[str, str | int]] = [] # stores the records in RAM after filtering out already processed records

        path = Path(input_path)
        if not path.is_file():
            raise FileNotFoundError(f"Input path {input_path} is not a valid file.")
        self.input_path = input_path
        self.output_path = output_path
        self.batch_size = batch_size
        self.input_file_rows = self.count_file_rows(input_path)
        self.max_rows = max_rows

    @staticmethod
    def count_file_rows(path: str) -> int:
        try:
            with open(path, "r") as f:
                reader = csv.reader(f)
                return sum(1 for row in reader) - 1 # subtract 1 for header row
        except FileNotFoundError:
            return 0
        
    def _add_already_processed_ids_to_set(self, already_processed_ids: set[str], output_files: list[str]):
        for output_file in output_files:
            try:
                with open(output_file, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        already_processed_ids.add(row["id"])
            except FileNotFoundError:
                continue

        return already_processed_ids

    def _return_already_processed_ids(self) -> set[str] | None:
        already_processed_ids = set()
        base_path = Path(self.output_path)

        # Ensure the directory exists to avoid errors
        if not base_path.is_dir():
            return already_processed_ids

        # rglob("*") searches recursively through all subdirectories
        # We filter for files named "metadata.json"
        output_files_to_check = []
        for metadata_file in base_path.rglob("metadata.json"):
            try:
                with open(metadata_file, "r") as f:
                    data = json.load(f)
                    
                    cli_args = data.get("cli_args", {})
                    if cli_args.get("input_path") == str(self.input_path):
                        output_files_to_check.append(metadata_file.parent / "output.csv")
                        
            except (json.JSONDecodeError, IOError, PermissionError):
                # Skip files that are empty, corrupted, or locked
                continue
        print(output_files_to_check)
        self._add_already_processed_ids_to_set(already_processed_ids, output_files_to_check)

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
                    print(f"Found {self.input_file_rows} rows. Processing first {self.max_rows} new rows...")
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
        """
        Used by tqdm to determine the total number of batches for the progress bar.
        """
        return math.ceil(len(self.data) / self.batch_size)