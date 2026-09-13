import sqlite3
from pathlib import Path

import torch

from llm.classifier import BusinessClassifier
from llm.classifier_dataset import (
    build_tokenizer,
    METRICS,
    GROUPS,
)
from llm.config import DEVICE

from semantic.loader import SemanticModel
from bql.compiler import BQLCompiler
from bql.models import Query

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "llm"
    / "business_classifier.pt"
)

SCHEMA_PATH = (
    PROJECT_ROOT
    / "semantic"
    / "schema.yaml"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "semantic"
    / "metrics.yaml"
)

DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "business.db"
)

def load_model():

    tokenizer = build_tokenizer()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    model = BusinessClassifier(
        vocab_size=tokenizer.vocab_size,
        num_metrics=len(METRICS),
        num_groups=len(GROUPS),
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model = model.to(DEVICE)
    model.eval()

    return model, tokenizer

def encode_question(
    question,
    tokenizer,
    max_length=64,
):

    tokens = tokenizer.encode(question)

    tokens = tokens[:max_length]

    pad_id = tokenizer.token_to_id["<PAD>"]

    if len(tokens) < max_length:

        tokens += [
            pad_id
        ] * (
            max_length - len(tokens)
        )

    return torch.tensor(
        [tokens],
        dtype=torch.long,
    )

def predict(
    question,
    model,
    tokenizer,
):

    device = torch.device(DEVICE)

    input_ids = encode_question(
        question,
        tokenizer,
    ).to(device)

    with torch.no_grad():

        metric_logits, group_logits = model(
            input_ids
        )

    metric_id = torch.argmax(
        metric_logits,
        dim=1,
    ).item()

    group_id = torch.argmax(
        group_logits,
        dim=1,
    ).item()

    metric = METRICS[metric_id]
    group = GROUPS[group_id]

    return metric, group

def build_bql(
    metric,
    group,
):

    if group == "none":

        return Query(
            metric=metric,
            group_by=[],
        )

    return Query(
        metric=metric,
        group_by=[group],
    )

def execute_sql(sql):

    connection = sqlite3.connect(DB_PATH)

    try:

        cursor = connection.cursor()

        cursor.execute(sql)

        rows = cursor.fetchall()

        columns = [
            description[0]
            for description in cursor.description
        ]

        return columns, rows

    finally:

        connection.close()

def main():

    print("=== Business Brain ===")
    print()

    model, tokenizer = load_model()

    question = input(
        "Question: "
    ).strip()

    if not question:

        print("No question provided.")
        return

    metric, group = predict(
        question,
        model,
        tokenizer,
    )

    print()
    print("Predicted metric:", metric)
    print("Predicted group:", group)

    query = build_bql(
        metric,
        group,
    )

    bql = {
        "metric": query.metric,
        "group_by": query.group_by,
    }

    print()
    print("Parsed BQL:")
    print(bql)

    try:

        semantic_model = SemanticModel(
            schema_path=str(SCHEMA_PATH),
            metrics_path=str(METRICS_PATH),
        )

    except Exception as exc:

        print()
        print("Semantic model loading failed:")
        print(exc)

        return

    try:

        compiler = BQLCompiler(
            semantic_model
        )

        sql = compiler.compile(
            query
        )

    except Exception as exc:

        print()
        print("BQL compilation failed:")
        print(exc)

        return

    print()
    print("Generated SQL:")
    print(sql)

    try:

        columns, rows = execute_sql(
            sql
        )

    except Exception as exc:

        print()
        print("SQL execution failed:")
        print(exc)

        return

    print()
    print("Results:")

    if not rows:

        print("No results found.")
        return

    print(
        " | ".join(columns)
    )

    print(
        "-+-".join(
            "-" * len(column)
            for column in columns
        )
    )

    for row in rows:

        print(
            " | ".join(
                str(value)
                for value in row
            )
        )

if __name__ == "__main__":
    main()
