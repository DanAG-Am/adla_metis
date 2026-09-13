import torch
import torch.nn as nn

class BusinessTransformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
        max_sequence_length=128,
        dropout=0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
        )

        self.position_embedding = nn.Embedding(
            max_sequence_length,
            embedding_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output = nn.Linear(
            embedding_dim,
            vocab_size,
        )

        self.max_sequence_length = max_sequence_length

    def forward(self, input_ids):

        batch_size, sequence_length = input_ids.shape

        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
        )

        positions = positions.unsqueeze(0).expand(
            batch_size,
            sequence_length,
        )

        x = (
            self.embedding(input_ids)
            + self.position_embedding(positions)
        )

        x = self.transformer(x)

        logits = self.output(x)

        return logits
