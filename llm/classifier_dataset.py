import torch
from torch.utils.data import Dataset

from llm.dataset import load_examples
from llm.tokenizer import SimpleTokenizer

METRICS = [
    "revenue",
    "orders",
    "units",
    "average_order_value",
]

GROUPS = [
    "none",
    "region",
    "segment",
    "category",
    "product",
]

def build_tokenizer():

    examples = load_examples()

    questions = [
        example["question"]
        for example in examples
    ]

    tokenizer = SimpleTokenizer()

    tokenizer.build_vocab(questions)

    return tokenizer

class BusinessIntentDataset(Dataset):

    def __init__(
        self,
        tokenizer,
        max_length=64,
    ):

        self.tokenizer = tokenizer
        self.max_length = max_length

        self.examples = load_examples()

    def __len__(self):

        return len(self.examples)

    def __getitem__(self, index):

        example = self.examples[index]

        question = example["question"]

        metric = example["bql"]["metric"]

        group_by = example["bql"]["group_by"]

        if group_by:

            group = group_by[0]

        else:

            group = "none"

        metric_id = METRICS.index(
            metric
        )

        group_id = GROUPS.index(
            group
        )

        tokens = self.tokenizer.encode(
            question
        )

        tokens = tokens[:self.max_length]

        padding_length = (
            self.max_length
            - len(tokens)
        )

        if padding_length > 0:

            tokens += [
                self.tokenizer.token_to_id[
                    "<PAD>"
                ]
            ] * padding_length

        return (
            torch.tensor(
                tokens,
                dtype=torch.long,
            ),
            torch.tensor(
                metric_id,
                dtype=torch.long,
            ),
            torch.tensor(
                group_id,
                dtype=torch.long,
            ),
        )

