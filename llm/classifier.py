import torch
import torch.nn as nn

class BusinessClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_metrics=4,
        num_groups=5,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0,
        )

        self.encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.metric_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_metrics),
        )

        self.group_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_groups),
        )

    def forward(self, input_ids):

        mask = input_ids != 0

        lengths = mask.sum(dim=1)

        embedded = self.embedding(input_ids)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden = self.encoder(packed)

        hidden = hidden[-1]

        metric_logits = self.metric_head(hidden)

        group_logits = self.group_head(hidden)

        return metric_logits, group_logits
