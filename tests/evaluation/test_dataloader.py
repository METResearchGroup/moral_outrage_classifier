from evaluation.dataloader import DataLoader
import pytest

class TestConstructor:
    def test_nonexistent_path(self):
        pass

class TestReturnAlreadyProcessedIds:
    def test_nonexistent_output_file(self):
        pass

    def test_empty_output_file(self):
        pass

    def test_nonempty_output_file(self):
        pass

class TestReturnNewRecords:
    def test_no_already_processed(self):
        pass

    def test_some_already_processed(self):
        pass

    def test_all_already_processed(self):
        pass

    def test_column_name_variations(self):
        pass

    def test_missing_gold_label(self):
        pass

class TestIter:
    def test_empty_data(self):
        pass

    def test_divisible_data_len(self):
        pass

    def test_undivisible_data_len(self):
        pass