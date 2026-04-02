import csv
from pathlib import Path
from typing import Iterator

column_name_conversion = {
    "id": ["id", "tweet_id"],
    "text": ["text", "body"],
    "gold_label": ["gold_label", "outrage", "pers_outrage_label"],
}


class DataLoader:
    def __init__(self, input_path: str, output_path: str, batch_size: int, max_rows: int | float = float('inf')):
        self.data: list[dict[str, str | int]] = []

        path = Path(input_path)
        if not path.is_file():
            raise FileNotFoundError(f"Input path {input_path} is not a valid file.")
        self.input_path = input_path
        self.output_path = output_path
        self.batch_size = batch_size
        self.max_rows = max_rows

        # prevent double labeling of texts ie same text appearing multiple times in input file
        self.texts = set()

    # puts all of the id's from output path into a set
    def _return_already_processed_ids(self) -> set[str]:
        already_processed_ids = set()
        try:
            with open(self.output_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    already_processed_ids.add(row["id"])
        except FileNotFoundError:
            # if output file doesn't exist yet, no records have been processed.
            pass
        return already_processed_ids
    
    def _append_new_data(self, row: dict[str, str], already_processed_ids: set[str], new_data: list[dict[str, str | int]]) -> None:
        id = next((row[key] for key in column_name_conversion["id"] if key in row), None)
        text = next((row[key] for key in column_name_conversion["text"] if key in row), None)
        if id not in already_processed_ids and text not in self.texts:
            gold_label_str = next((row[key] for key in column_name_conversion["gold_label"] if key in row), None)
            gold_label = int(gold_label_str) if gold_label_str is not None else None
            new_data.append({"text": text, "gold_label": gold_label, "id": id})
            self.texts.add(text)

    # use the set of already processed id's to filter out records from input path
    def _return_new_records(self, already_processed_ids: set[str]) -> list[dict[str, str | int]]:
        new_data = []
        with open(self.input_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._append_new_data(row, already_processed_ids, new_data)
                if len(self.texts) >= self.max_rows:
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