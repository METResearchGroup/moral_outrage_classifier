import json
from pathlib import Path

import pytest

from evaluation.dataloader import DataLoader
from evaluation.metadata import get_model_id_value
from models.llm.models import AnthropicModel, OpenAIModel


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
    mid = get_model_id_value("perspective_api")
    f.write_text(
        f"id,dataset,text,gold_label,model,pred_label,is_correct\n"
        f"1,{input_file_with_rows},hello world,0,{mid},0,1\n"
    )
    return f

@pytest.fixture
def output_file_with_all_rows(tmp_path, input_file_with_rows):
    f = tmp_path / "output.csv"
    mid = get_model_id_value("perspective_api")
    f.write_text(
        f"id,dataset,text,gold_label,model,pred_label,is_correct\n"
        f"1,{input_file_with_rows},hello world,0,{mid},0,1\n"
        f"2,{input_file_with_rows},this is outrageous,1,{mid},1,1\n"
    )
    return f


def _openai_metadata_entry(
    *,
    llm_provider_name: str = "openai",
    resolved_model_id: str | None = None,
    prompt_hash: str | None = None,
) -> dict:
    rid = resolved_model_id if resolved_model_id is not None else OpenAIModel.get_resolved_model_id()
    ph = prompt_hash if prompt_hash is not None else OpenAIModel.get_prompt_hash()
    return {
        "llm_provider_name": llm_provider_name,
        "resolved_model_id": rid,
        "prompt_hash": ph,
        "prompt_template": OpenAIModel.get_prompt_template(),
    }


def _perspective_metadata_entry() -> dict:
    alias = "perspective_api"
    return {
        "llm_provider_name": alias,
        "resolved_model_id": get_model_id_value(alias),
        "prompt_hash": None,
        "prompt_template": None,
    }


def _write_run_artifacts(
    run_dir: Path,
    input_csv: Path,
    models_meta: list[dict],
    output_csv_lines: list[str],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"cli_args": {"input_path": str(input_csv)}, "models": models_meta}
    (run_dir / "metadata.json").write_text(json.dumps(metadata))
    header = "id,dataset,text,gold_label,pred_label,is_correct,model\n"
    (run_dir / "output.csv").write_text(header + "".join(output_csv_lines))


class TestConstructor:
    """Invalid input path raises before any I/O on output."""

    def test_nonexistent_path(self, nonexistent_output_file, tmp_path):
        with pytest.raises(FileNotFoundError):
            DataLoader(input_path=str(tmp_path / "nonexistent.csv"), output_path=str(nonexistent_output_file), batch_size=10, model_name="perspective_api")


class TestReturnAlreadyProcessedIds:
    """Covers `_return_already_processed_ids` (output root missing or not a directory → empty set)."""

    def test_nonexistent_output_dir_returns_empty_set(self, input_file_with_rows, nonexistent_output_file):
        loader = DataLoader(str(input_file_with_rows), str(nonexistent_output_file), batch_size=10, model_name="perspective_api")
        assert loader._return_already_processed_ids() == set()


class TestGetUniqueModelRunIdentifier:
    """Matches `DataLoader.get_unique_model_run_identifier` (resolved_model_id + prompt_hash for LLM; alias + None for Perspective)."""

    def test_openai_returns_resolved_id_and_prompt_hash(self, input_file_with_rows, tmp_path):
        loader = DataLoader(str(input_file_with_rows), str(tmp_path), batch_size=10, model_name="openai")
        resolved_id, prompt_hash = loader.get_unique_model_run_identifier()
        assert resolved_id == get_model_id_value("openai")
        assert prompt_hash == OpenAIModel.get_prompt_hash()

    def test_anthropic_returns_resolved_id_and_prompt_hash(self, input_file_with_rows, tmp_path):
        loader = DataLoader(str(input_file_with_rows), str(tmp_path), batch_size=10, model_name="anthropic")
        resolved_id, prompt_hash = loader.get_unique_model_run_identifier()
        assert resolved_id == get_model_id_value("anthropic")
        assert resolved_id == AnthropicModel.get_resolved_model_id()
        assert prompt_hash == AnthropicModel.get_prompt_hash()

    def test_perspective_returns_alias_and_no_prompt_hash(self, input_file_with_rows, tmp_path):
        loader = DataLoader(str(input_file_with_rows), str(tmp_path), batch_size=10, model_name="perspective_api")
        resolved_id, prompt_hash = loader.get_unique_model_run_identifier()
        assert resolved_id == get_model_id_value("perspective_api")
        assert prompt_hash is None


class TestLoadMetadataFilesFromPastDuplicateRuns:
    """Covers `_load_metadata_files_from_past_duplicate_runs` (paths to prior `output.csv` that match input + metadata identity)."""

    def test_returns_matching_run_output_csv(self, tmp_path, input_file_with_three_rows):
        out_root = tmp_path / "outputs"
        ds = str(input_file_with_three_rows)
        row = f"1,{ds},hello world,0,0,1,{get_model_id_value('openai')}\n"
        _write_run_artifacts(
            out_root / "run1",
            input_file_with_three_rows,
            [_openai_metadata_entry()],
            [row],
        )
        loader = DataLoader(ds, str(out_root), batch_size=10, model_name="openai")
        rid, ph = loader.get_unique_model_run_identifier()
        paths = loader._load_metadata_files_from_past_duplicate_runs(rid, ph)
        assert paths == [out_root / "run1" / "output.csv"]

    def test_returns_empty_when_metadata_models_do_not_match_identity(
        self, tmp_path, input_file_with_three_rows
    ):
        out_root = tmp_path / "outputs"
        ds = str(input_file_with_three_rows)
        _write_run_artifacts(
            out_root / "run1",
            input_file_with_three_rows,
            [_openai_metadata_entry(resolved_model_id="other-model")],
            [f"1,{ds},hello world,0,0,1,other-model\n"],
        )
        loader = DataLoader(ds, str(out_root), batch_size=10, model_name="openai")
        rid, ph = loader.get_unique_model_run_identifier()
        assert loader._load_metadata_files_from_past_duplicate_runs(rid, ph) == []


class TestReturnNewRecords:
    """Covers `_return_new_records` (filter input CSV by already-processed id set)."""

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
    """Batches from `load_data` for tqdm / harness consumption."""

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

    def test_indivisible_data_len(self, input_file_with_three_rows, nonexistent_output_file):
        loader = DataLoader(str(input_file_with_three_rows), str(nonexistent_output_file), batch_size=2, model_name="perspective_api")
        loader.load_data()
        batches = list(loader)
        assert len(batches) == 2
        assert len(batches[0]) == 2
        assert len(batches[1]) == 1


class TestResumeDedupUsesRunIdentifierMetadataAndModelIdColumn:
    """End-to-end resume: prior `metadata.json` `models[]` must match `get_unique_model_run_identifier`; rows reused only when `model` equals `get_model_id_value`."""

    def test_same_input_path_same_run_identifier_skips_already_processed_rows(
        self, tmp_path, input_file_with_three_rows
    ):
        out_root = tmp_path / "outputs"
        ds = str(input_file_with_three_rows)
        row = f"1,{ds},hello world,0,0,1,{get_model_id_value('openai')}\n"
        _write_run_artifacts(
            out_root / "run1",
            input_file_with_three_rows,
            [_openai_metadata_entry()],
            [row],
        )
        loader = DataLoader(ds, str(out_root), batch_size=10, model_name="openai")
        loader.load_data()
        ids = [r["id"] for r in loader.data]
        assert ids == ["2", "3"]

    def test_different_llm_provider_name_same_run_identifier_still_skips_rows(
        self, tmp_path, input_file_with_three_rows
    ):
        """Resume matches `metadata.models[]` on resolved_model_id + prompt_hash, not llm_provider_name."""
        out_root = tmp_path / "outputs"
        ds = str(input_file_with_three_rows)
        row = f"1,{ds},hello world,0,0,1,{get_model_id_value('openai')}\n"
        _write_run_artifacts(
            out_root / "run1",
            input_file_with_three_rows,
            [_openai_metadata_entry(llm_provider_name="other_alias")],
            [row],
        )
        loader = DataLoader(ds, str(out_root), batch_size=10, model_name="openai")
        loader.load_data()
        assert [r["id"] for r in loader.data] == ["2", "3"]

    def test_metadata_resolved_model_mismatch_does_not_skip_rows(
        self, tmp_path, input_file_with_three_rows
    ):
        out_root = tmp_path / "outputs"
        ds = str(input_file_with_three_rows)
        old_resolved = "legacy-model-id"
        row = f"1,{ds},hello world,0,0,1,{old_resolved}\n"
        _write_run_artifacts(
            out_root / "run1",
            input_file_with_three_rows,
            [_openai_metadata_entry(resolved_model_id=old_resolved)],
            [row],
        )
        loader = DataLoader(ds, str(out_root), batch_size=10, model_name="openai")
        loader.load_data()
        assert [r["id"] for r in loader.data] == ["1", "2", "3"]

    def test_metadata_prompt_hash_mismatch_does_not_skip_rows(
        self, tmp_path, input_file_with_three_rows
    ):
        out_root = tmp_path / "outputs"
        ds = str(input_file_with_three_rows)
        wrong_hash = "0" * 64
        row = f"1,{ds},hello world,0,0,1,{get_model_id_value('openai')}\n"
        _write_run_artifacts(
            out_root / "run1",
            input_file_with_three_rows,
            [_openai_metadata_entry(prompt_hash=wrong_hash)],
            [row],
        )
        loader = DataLoader(ds, str(out_root), batch_size=10, model_name="openai")
        loader.load_data()
        assert [r["id"] for r in loader.data] == ["1", "2", "3"]

    def test_perspective_output_rows_do_not_skip_llm_rows_for_openai_loader(
        self, tmp_path, input_file_with_three_rows
    ):
        out_root = tmp_path / "outputs"
        ds = str(input_file_with_three_rows)
        lines = [
            f"1,{ds},hello world,0,0,1,{get_model_id_value('perspective_api')}\n",
            f"2,{ds},this is outrageous,1,1,1,{get_model_id_value('openai')}\n",
        ]
        _write_run_artifacts(
            out_root / "run1",
            input_file_with_three_rows,
            [_openai_metadata_entry(), _perspective_metadata_entry()],
            lines,
        )
        loader = DataLoader(ds, str(out_root), batch_size=10, model_name="openai")
        loader.load_data()
        assert [r["id"] for r in loader.data] == ["1", "3"]

    def test_llm_output_rows_do_not_skip_perspective_rows_for_perspective_loader(
        self, tmp_path, input_file_with_three_rows
    ):
        out_root = tmp_path / "outputs"
        ds = str(input_file_with_three_rows)
        lines = [
            f"1,{ds},hello world,0,0,1,{get_model_id_value('openai')}\n",
            f"2,{ds},this is outrageous,1,1,1,{get_model_id_value('perspective_api')}\n",
        ]
        _write_run_artifacts(
            out_root / "run1",
            input_file_with_three_rows,
            [_openai_metadata_entry(), _perspective_metadata_entry()],
            lines,
        )
        loader = DataLoader(ds, str(out_root), batch_size=10, model_name="perspective_api")
        loader.load_data()
        assert [r["id"] for r in loader.data] == ["1", "3"]

    def test_missing_models_array_in_metadata_skips_run_for_resume_dedup(
        self, tmp_path, input_file_with_rows
    ):
        out_root = tmp_path / "outputs"
        run_dir = out_root / "legacy_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        ds = str(input_file_with_rows)
        (run_dir / "metadata.json").write_text(
            json.dumps({"cli_args": {"input_path": ds}})
        )
        (run_dir / "output.csv").write_text(
            f"id,dataset,text,gold_label,pred_label,is_correct,model\n"
            f"1,{ds},hello world,0,0,1,{get_model_id_value('perspective_api')}\n"
        )
        loader = DataLoader(ds, str(out_root), batch_size=10, model_name="perspective_api")
        loader.load_data()
        assert [r["id"] for r in loader.data] == ["1", "2"]
