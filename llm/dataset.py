import json

import torch
from torch.utils.data import Dataset

def load_examples():

    examples = []

    with open(
        "training_data/examples.jsonl",
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            examples.append(
                json.loads(line)
            )

    return examples

def bql_to_target(bql):

    metric = bql["metric"]

    group_by = bql.get(
        "group_by",
        [],
    )

    if group_by:

        return (
            metric
            + "|"
            + ",".join(group_by)
        )

    return metric + "|"

def example_to_text(example):

    question = example["question"]

    target = bql_to_target(
        example["bql"]
    )

    return (
        question
        + " <SEP> "
        + target
    )

class BusinessDataset(Dataset):

    def __init__(
        self,
        tokenizer,
        max_length=128,
    ):

        self.tokenizer = tokenizer

        self.max_length = max_length

        self.examples = load_examples()

        self.data = []

        for example in self.examples:

            question = example["question"]

            target = bql_to_target(
                example["bql"]
            )

            question_ids = tokenizer.encode(
                question,
                add_special_tokens=False,
            )

            target_ids = tokenizer.encode(
                target,
                add_special_tokens=False,
            )

            bos_id = tokenizer.token_to_id[
                "<BOS>"
            ]

            sep_id = tokenizer.token_to_id[
                "<SEP>"
            ]

            eos_id = tokenizer.token_to_id[
                "<EOS>"
            ]

            input_ids = (
                [bos_id]
                + question_ids
                + [sep_id]
                + target_ids
                + [eos_id]
            )

            labels = (
                [-100]
                * (
                    1
                    + len(question_ids)
                    + 1
                )
                + target_ids
                + [eos_id]
            )

            input_ids = input_ids[
                :max_length
            ]

            labels = labels[
                :max_length
            ]

            padding_length = (
                max_length
                - len(input_ids)
            )

            if padding_length > 0:

                pad_id = tokenizer.token_to_id[
                    "<PAD>"
                ]

                input_ids += (
                    [pad_id]
                    * padding_length
                )

                labels += (
                    [-100]
                    * padding_length
                )

            self.data.append(
                (
                    torch.tensor(
                        input_ids,
                        dtype=torch.long,
                    ),
                    torch.tensor(
                        labels,
                        dtype=torch.long,
                    ),
                )
            )

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        return self.data[index]
