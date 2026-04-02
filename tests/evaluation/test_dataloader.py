from evaluation.dataloader import DataLoader
import pytest


# Input file fixtures
@pytest.fixture
def input_file_with_headers(tmp_path):
    f = tmp_path / "input.csv"
    f.write_text(
        "id,text,gold_label\n"
    )
    return f

@pytest.fixture
def input_file_with_rows(tmp_path):
    f = tmp_path / "input.csv"
    f.write_text(
        "id,text,gold_label\n"
        "1,hello world,0\n"
        "2,this is outrageous,1\n"
    )
    return f

@pytest.fixture
def input_file_with_three_rows(tmp_path):
    f = tmp_path / "input.csv"
    f.write_text(
        "id,text,gold_label\n"
        "1,hello world,0\n"
        "2,this is outrageous,1\n"
        "3,so angry,1\n"
    )
    return f

# column header variation input files
@pytest.fixture
def input_file_with_tweet_id_column(tmp_path):
    f = tmp_path / "input.csv"
    f.write_text(
        "tweet_id,text,gold_label\n"
        "1,hello world,0\n"
        "2,this is outrageous,1\n"
    )
    return f

@pytest.fixture
def input_file_with_body_column(tmp_path):
    f = tmp_path / "input.csv"
    f.write_text(
        "id,body,gold_label\n"
        "1,hello world,0\n"
        "2,this is outrageous,1\n"
    )
    return f

@pytest.fixture
def input_file_with_outrage_column(tmp_path):
    f = tmp_path / "input.csv"
    f.write_text(
        "id,text,outrage\n"
        "1,hello world,0\n"
        "2,this is outrageous,1\n"
    )
    return f

@pytest.fixture
def input_file_with_pers_outrage_label_column(tmp_path):
    f = tmp_path / "input.csv"
    f.write_text(
        "id,text,pers_outrage_label\n"
        "1,hello world,0\n"
        "2,this is outrageous,1\n"
    )
    return f

@pytest.fixture
def input_file_with_all_alternative_columns(tmp_path):
    f = tmp_path / "input.csv"
    f.write_text(
        "tweet_id,body,pers_outrage_label\n"
        "1,hello world,0\n"
        "2,this is outrageous,1\n"
    )
    return f


@pytest.fixture(params=[
    "input_file_with_tweet_id_column",
    "input_file_with_body_column",
    "input_file_with_outrage_column",
    "input_file_with_pers_outrage_label_column",
    "input_file_with_all_alternative_columns",
])
def column_variation_input(request):
    return request.getfixturevalue(request.param)


# Output file fixtures
@pytest.fixture
def empty_output_file(tmp_path):
    f = tmp_path / "output.csv"
    f.write_text("")
    return f

@pytest.fixture
def nonexistent_output_file(tmp_path):
    return tmp_path / "output.csv"

@pytest.fixture
def output_file_with_rows(tmp_path, input_file_with_rows):
    f = tmp_path / "output.csv"
    f.write_text(
        f"id,dataset,text,gold_label,model,pred_label,is_correct\n"
        f"1,{input_file_with_rows},hello world,0,perspective_api,0,1\n"
    )
    return f

@pytest.fixture
def output_file_with_all_rows(tmp_path, input_file_with_rows):
    f = tmp_path / "output.csv"
    f.write_text(
        f"id,dataset,text,gold_label,model,pred_label,is_correct\n"
        f"1,{input_file_with_rows},hello world,0,perspective_api,0,1\n"
        f"2,{input_file_with_rows},this is outrageous,1,perspective_api,1,1\n"
    )
    return f


class TestConstructor:
    def test_nonexistent_path(self, nonexistent_output_file, tmp_path):
        with pytest.raises(FileNotFoundError):
            DataLoader(input_path=str(tmp_path / "nonexistent.csv"), output_path=str(nonexistent_output_file), batch_size=10, model_name="perspective_api")


class TestReturnAlreadyProcessedIds:
    def test_nonexistent_output_file(self, input_file_with_rows, nonexistent_output_file):
        loader = DataLoader(str(input_file_with_rows), str(nonexistent_output_file), batch_size=10, model_name="perspective_api")
        assert loader._return_already_processed_ids() == set()

    def test_empty_output_file(self, input_file_with_rows, empty_output_file):
        loader = DataLoader(str(input_file_with_rows), str(empty_output_file), batch_size=10, model_name="perspective_api")
        assert loader._return_already_processed_ids() == set()

    def test_nonempty_output_file(self, input_file_with_rows, output_file_with_rows):
        loader = DataLoader(str(input_file_with_rows), str(output_file_with_rows), batch_size=10, model_name="perspective_api")
        assert loader._return_already_processed_ids() == {"1"}


class TestReturnNewRecords:
    def test_no_already_processed(self, input_file_with_rows, nonexistent_output_file):
        loader = DataLoader(str(input_file_with_rows), str(nonexistent_output_file), batch_size=10, model_name="perspective_api")
        result = loader._return_new_records(set())
        assert len(result) == 2
        assert result[0] == {"id": "1", "text": "hello world", "gold_label": 0}
        assert result[1] == {"id": "2", "text": "this is outrageous", "gold_label": 1}

    def test_some_already_processed(self, input_file_with_rows, output_file_with_rows):
        loader = DataLoader(str(input_file_with_rows), str(output_file_with_rows), batch_size=10, model_name="perspective_api")
        result = loader._return_new_records({"1"})
        assert len(result) == 1
        assert result[0]["id"] == "2"

    def test_all_already_processed(self, input_file_with_rows, output_file_with_all_rows):
        loader = DataLoader(str(input_file_with_rows), str(output_file_with_all_rows), batch_size=10, model_name="perspective_api")
        result = loader._return_new_records({"1", "2"})
        assert result == []

    def test_column_name_variations(self, column_variation_input, nonexistent_output_file):
        loader = DataLoader(str(column_variation_input), str(nonexistent_output_file), batch_size=10, model_name="perspective_api")
        result = loader._return_new_records(set())
        assert len(result) == 2
        assert result[0]["text"] == "hello world"
        assert result[0]["gold_label"] == 0


class TestIter:
    def test_empty_data(self, input_file_with_headers, nonexistent_output_file):
        loader = DataLoader(str(input_file_with_headers), str(nonexistent_output_file), batch_size=10, model_name="perspective_api")
        loader.load_data()
        assert list(loader) == []

    def test_divisible_data_len(self, input_file_with_rows, nonexistent_output_file):
        loader = DataLoader(str(input_file_with_rows), str(nonexistent_output_file), batch_size=2, model_name="perspective_api")
        loader.load_data()
        batches = list(loader)
        assert len(batches) == 1
        assert len(batches[0]) == 2

    def test_undivisible_data_len(self, input_file_with_three_rows, nonexistent_output_file):
        loader = DataLoader(str(input_file_with_three_rows), str(nonexistent_output_file), batch_size=2, model_name="perspective_api")
        loader.load_data()
        batches = list(loader)
        assert len(batches) == 2
        assert len(batches[0]) == 2
        assert len(batches[1]) == 1
