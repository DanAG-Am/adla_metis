import torch

from llm.config import DEVICE, MODEL_PATH
from llm.model import BusinessTransformer
from llm.tokenizer import SimpleTokenizer

VALID_METRICS = [
    "revenue",
    "orders",
    "units",
    "average_order_value",
]

VALID_GROUPS = [
    "region",
    "segment",
    "category",
    "product",
]

def load_model():

    device = torch.device(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    tokenizer = SimpleTokenizer()

    tokenizer.token_to_id = checkpoint[
        "token_to_id"
    ]

    tokenizer.id_to_token = {
        int(key): value
        for key, value in checkpoint[
            "id_to_token"
        ].items()
    }

    model = BusinessTransformer(
        vocab_size=len(
            tokenizer.token_to_id
        ),
        max_sequence_length=checkpoint[
            "max_sequence_length"
        ],
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model = model.to(device)

    model.eval()

    return model, tokenizer, device

def get_id(tokenizer, token):

    return tokenizer.token_to_id.get(token)

def allowed_first_tokens(tokenizer):

    ids = []

    for metric in VALID_METRICS:

        token_id = get_id(
            tokenizer,
            metric,
        )

        if token_id is not None:

            ids.append(token_id)

    return ids

def allowed_group_tokens(tokenizer):

    ids = []

    for group in VALID_GROUPS:

        token_id = get_id(
            tokenizer,
            group,
        )

        if token_id is not None:

            ids.append(token_id)

    return ids

def choose_token(
    logits,
    allowed_ids,
):

    if not allowed_ids:

        raise ValueError(
            "No valid tokens available."
        )

    allowed_logits = logits[
        allowed_ids
    ]

    best_index = torch.argmax(
        allowed_logits
    ).item()

    return allowed_ids[best_index]

def generate(
    model,
    tokenizer,
    device,
    question,
):

    bos_id = tokenizer.token_to_id[
        "<BOS>"
    ]

    sep_id = tokenizer.token_to_id[
        "<SEP>"
    ]

    eos_id = tokenizer.token_to_id[
        "<EOS>"
    ]

    pipe_id = tokenizer.token_to_id.get(
        "|"
    )

    if pipe_id is None:

        raise ValueError(
            "Tokenizer does not contain '|'."
        )

    question_tokens = tokenizer.encode(
        question,
        add_special_tokens=False,
    )

    token_ids = (
        [bos_id]
        + question_tokens
        + [sep_id]
    )

    generated_tokens = []

    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():

        logits = model(input_ids)

    next_token_id = choose_token(
        logits[0, -1],
        allowed_first_tokens(
            tokenizer
        ),
    )

    generated_tokens.append(
        next_token_id
    )

    token_ids.append(
        next_token_id
    )

    generated_tokens.append(
        pipe_id
    )

    token_ids.append(
        pipe_id
    )

    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():

        logits = model(input_ids)

    next_logits = logits[0, -1]

    group_ids = allowed_group_tokens(
        tokenizer
    )

    candidates = group_ids + [eos_id]

    next_token_id = choose_token(
        next_logits,
        candidates,
    )

    if next_token_id == eos_id:

        generated_tokens.append(
            eos_id
        )

        return tokenizer.decode(
            generated_tokens
        )

    generated_tokens.append(
        next_token_id
    )

    return tokenizer.decode(
        generated_tokens
    )

def parse_bql(text):

    text = text.strip()

    if "|" not in text:

        raise ValueError(
            f"Invalid BQL output: {text}"
        )

    metric, group = text.split(
        "|",
        1,
    )

    metric = metric.strip()

    group = group.strip()

    if metric not in VALID_METRICS:

        raise ValueError(
            f"Unknown metric: {metric}"
        )

    if group:

        if group not in VALID_GROUPS:

            raise ValueError(
                f"Unknown group: {group}"
            )

        group_by = [group]

    else:

        group_by = []

    return {
        "metric": metric,
        "group_by": group_by,
    }

def main():

    model, tokenizer, device = load_model()

    question = input(
        "Question: "
    ).strip()

    generated = generate(
        model,
        tokenizer,
        device,
        question,
    )

    print()
    print("Generated intent:")
    print(generated)

    try:

        bql = parse_bql(
            generated
        )

        print()
        print("Parsed BQL:")
        print(bql)

    except ValueError as error:

        print()
        print("BQL parsing failed:")
        print(error)

if __name__ == "__main__":
    main()
