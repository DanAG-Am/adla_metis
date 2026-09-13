import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from llm.classifier import BusinessClassifier
from llm.classifier_dataset import (
    build_tokenizer,
    BusinessIntentDataset,
    METRICS,
    GROUPS,
)

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

EPOCHS = 1000
BATCH_SIZE = 8
LEARNING_RATE = 0.003

MODEL_PATH = "llm/business_classifier.pt"

def main():

    print()
    print("=== Business Brain Classifier Training ===")
    print()

    device = torch.device(DEVICE)

    print("Device:", device)

    tokenizer = build_tokenizer()

    dataset = BusinessIntentDataset(
        tokenizer,
        max_length=64,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    print("Training examples:", len(dataset))
    print("Vocabulary size:", tokenizer.vocab_size)
    print("Metrics:", METRICS)
    print("Groups:", GROUPS)

    model = BusinessClassifier(
        vocab_size=tokenizer.vocab_size,
        num_metrics=len(METRICS),
        num_groups=len(GROUPS),
    ).to(device)

    print(
        "Model parameters:",
        sum(p.numel() for p in model.parameters())
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )

    metric_loss_fn = nn.CrossEntropyLoss()
    group_loss_fn = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(EPOCHS):

        total_loss = 0.0

        for inputs, metric_targets, group_targets in dataloader:

            inputs = inputs.to(device)
            metric_targets = metric_targets.to(device)
            group_targets = group_targets.to(device)

            optimizer.zero_grad()

            metric_logits, group_logits = model(inputs)

            metric_loss = metric_loss_fn(
                metric_logits,
                metric_targets,
            )

            group_loss = group_loss_fn(
                group_logits,
                group_targets,
            )

            loss = metric_loss + group_loss

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        model.eval()

        correct_metric = 0
        correct_group = 0
        total = 0

        with torch.no_grad():

            for inputs, metric_targets, group_targets in dataloader:

                inputs = inputs.to(device)
                metric_targets = metric_targets.to(device)
                group_targets = group_targets.to(device)

                metric_logits, group_logits = model(inputs)

                metric_predictions = metric_logits.argmax(dim=1)
                group_predictions = group_logits.argmax(dim=1)

                correct_metric += (
                    metric_predictions == metric_targets
                ).sum().item()

                correct_group += (
                    group_predictions == group_targets
                ).sum().item()

                total += inputs.size(0)

        model.train()

        metric_accuracy = (
            correct_metric / total
        )

        group_accuracy = (
            correct_group / total
        )

        average_loss = (
            total_loss / len(dataloader)
        )

        if (
            epoch == 0
            or (epoch + 1) % 25 == 0
            or (
                metric_accuracy == 1.0
                and group_accuracy == 1.0
            )
        ):

            print(
                f"Epoch {epoch + 1:4d}/{EPOCHS} "
                f"Loss: {average_loss:.4f} "
                f"Metric Acc: {metric_accuracy:.2%} "
                f"Group Acc: {group_accuracy:.2%}"
            )

        if (
            metric_accuracy == 1.0
            and group_accuracy == 1.0
        ):

            print()
            print("100% training accuracy reached.")
            print("Classifier successfully learned the training set.")

            break

    torch.save(
        {
            "model_state": model.state_dict(),
            "token_to_id": tokenizer.token_to_id,
            "id_to_token": tokenizer.id_to_token,
            "metrics": METRICS,
            "groups": GROUPS,
        },
        MODEL_PATH,
    )

    print()
    print("Training complete.")
    print("Model saved to:")
    print(MODEL_PATH)

if __name__ == "__main__":
    main()
