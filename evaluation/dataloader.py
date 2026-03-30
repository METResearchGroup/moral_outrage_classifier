class DataLoader:
    def __init__(self, input_path: str, output_path: str, batch_size: int):
        self.data: list[tuple[str, int]] = []
        self.input_path = input_path
        self.output_path = output_path
        self.batch_size = batch_size

    def load_data(self):
        # deal with all the steps related to loading the data. Should save the resulting output to `self.data` in (input, output) pairs, where input = the text string, output = label (0/1) for moral outrage.
        # also should manage filtering out "already-processed" IDs. This way, EvaluationHarness can assume that loaded data is already filtered out.
        self.filter_already_processed_records()

        # iterate through input file in batch, if the id of the record is not in the set of already processed records, then we keep the record and add it to self.data. Otherwise, we skip the record. 

    def filter_already_processed_records(self):
        pass
        # declare and initialize a set storing all rows in output path
        # read output file in batch, and puts all id's into a set
        # iterate through input file in batch, and only keep those records whose id is not in the set of already processed records.
        # return the id's of records that are not in the set of already processed records.

    def __iter__(self) -> list[tuple[str, int]]:
        for i in range(0, len(self.data), self.batch_size):
            yield self.data[i:i + self.batch_size]
          # review "dunder methods" and "__iter__" on how to define iterators. This method should iterate the self.data to get a batch of  [(input, output)] pairs during iteration. For example, if batch size = 10, we should have [(x_1, y_1), ..., (x_10, y_10)]. Make sure to make this also robust if the sample size is not a multiple of the batch size (e.g., you have 9 training samples but a batch size of 10, or if you have 21 training samples but a batch size of 10). Also look up generators in Python and the difference between return vs. yield.