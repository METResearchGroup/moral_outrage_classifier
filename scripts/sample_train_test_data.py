"""One-off script to create a random sample of training/test data from the one-off script."""

TOTAL_TRAIN_SAMPLES = 100
TOTAL_TEST_SAMPLES = 100

from lib.constants import REPO_ROOT
from lib.timestamp_utils import get_current_timestamp

import pandas as pd

timestamp = get_current_timestamp()
TOTAL_DATA_FILEPATH = REPO_ROOT / "evaluation" / "sample_data" / "26k_training_data.csv"
OUTPUT_DATA_DIRPATH = REPO_ROOT / "evaluation" / "sample_data" / timestamp
TRAIN_DATA_FILEPATH = OUTPUT_DATA_DIRPATH / "train_data.csv"
TEST_DATA_FILEPATH = OUTPUT_DATA_DIRPATH / "test_data.csv"

def main():

    # load init data
    total_data = pd.read_csv(TOTAL_DATA_FILEPATH)

    # randomly sample for train/test (without replacement)
    total_sampled_data = total_data.sample(n=TOTAL_TRAIN_SAMPLES + TOTAL_TEST_SAMPLES, random_state=42)
    train_data = total_sampled_data.iloc[:TOTAL_TRAIN_SAMPLES]
    test_data = total_sampled_data.iloc[TOTAL_TRAIN_SAMPLES:]

    total_sampled_training = len(train_data)
    total_sampled_test = len(test_data)

    print(f"Total sampled training data: {total_sampled_training}")
    print(f"Total sampled test data: {total_sampled_test}")

    # save train/test data
    OUTPUT_DATA_DIRPATH.mkdir(parents=True, exist_ok=True)
    train_data.to_csv(TRAIN_DATA_FILEPATH, index=False)
    test_data.to_csv(TEST_DATA_FILEPATH, index=False)

if __name__ == "__main__":
    main()
