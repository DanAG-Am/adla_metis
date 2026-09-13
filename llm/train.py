import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from llm.config import (
    DEVICE,
    EMBEDDING_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    DROPOUT,
    MAX_SEQUENCE_LENGTH,
    LEARNING_RATE,
    EPOCHS,
    MODEL_PATH,
)

from llm.dataset import (
    load_examples,
    example_to_text,
    BusinessDataset,
)

from llm.model import BusinessTransformer
from llm.tokenizer import SimpleTokenizer

BATCH_SIZE = 4

def build_tokenizer():

    examples = load_examples()

    texts = [
        example_to_text(example)
        for example in examples
    ]

    tokenizer = SimpleTokenizer()

    tokenizer.build_vocab(texts)

    return tokenizer

def main():

    print("=== Business Brain Training ===")
    print()

    print("Device:", DEVICE)

    device = torch.device(DEVICE)

    tokenizer = build_tokenizer()

    print(
        "Vocabulary size:",
        tokenizer.vocab_size,
    )

    dataset = BusinessDataset(
        tokenizer,
        max_length=MAX_SEQUENCE_LENGTH,
    )

    print(
        "Training examples:",
        len(dataset),
    )

    print(
        "Sequence length:",
        dataset.max_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = BusinessTransformer(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        dropout=DROPOUT,
    )

    model = model.to(device)

    print(
        "Model parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        ),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    loss_function = nn.CrossEntropyLoss(
        ignore_index=-100
    )

    model.train()

    for epoch in range(EPOCHS):

        total_loss = 0.0

        for inputs, targets in dataloader:

            inputs = inputs.to(device)

            targets = targets.to(device)

            optimizer.zero_grad()

            logits = model(inputs)

            loss = loss_function(
                logits.reshape(
                    -1,
                    tokenizer.vocab_size,
                ),
                targets.reshape(-1),
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss
            / len(dataloader)
        )

        if (
            epoch == 0
            or (epoch + 1) % 10 == 0
        ):

            print(
                f"Epoch {epoch + 1:3d}/{EPOCHS} "
                f"Loss: {average_loss:.4f}"
            )

    torch.save(
        {
            "model_state": model.state_dict(),
            "token_to_id": tokenizer.token_to_id,
            "id_to_token": tokenizer.id_to_token,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
        },
        MODEL_PATH,
    )

    print()
    print("Training complete.")
    print("Model saved to:")
    print(MODEL_PATH)

if __name__ == "__main__":
    main()
