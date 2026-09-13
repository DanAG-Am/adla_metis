import torch

from llm.model import BusinessTransformer

def main():

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Device:", device)

    vocab_size = 100

    model = BusinessTransformer(
        vocab_size=vocab_size,
    )

    model = model.to(device)

    input_ids = torch.randint(
        0,
        vocab_size,
        (2, 20),
        device=device,
    )

    logits = model(input_ids)

    print("Input shape:")
    print(input_ids.shape)

    print()

    print("Output shape:")
    print(logits.shape)

    print()

    print(
        "Parameters:",
        sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
    )

if __name__ == "__main__":
    main()
