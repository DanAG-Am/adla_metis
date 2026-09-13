import re

SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
    "<SEP>",
]

class SimpleTokenizer:

    def __init__(self):

        self.token_to_id = {}
        self.id_to_token = {}

        for token in SPECIAL_TOKENS:
            self._add_token(token)

    def _add_token(self, token):

        if token not in self.token_to_id:

            token_id = len(self.token_to_id)

            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token

    def tokenize(self, text):

        tokens = re.findall(
            r"<(?:PAD|UNK|BOS|EOS|SEP)>|\w+|[^\w\s]",
            text,
            flags=re.IGNORECASE,
        )

        normalized = []

        for token in tokens:

            if token.upper() in SPECIAL_TOKENS:

                normalized.append(
                    token.upper()
                )

            else:

                normalized.append(
                    token.lower()
                )

        return normalized

    def build_vocab(self, texts):

        for text in texts:

            tokens = self.tokenize(text)

            for token in tokens:
                self._add_token(token)

    def encode(
        self,
        text,
        add_special_tokens=True,
    ):

        tokens = self.tokenize(text)

        if add_special_tokens:

            tokens = (
                ["<BOS>"]
                + tokens
                + ["<EOS>"]
            )

        return [
            self.token_to_id.get(
                token,
                self.token_to_id["<UNK>"],
            )
            for token in tokens
        ]

    def decode(self, token_ids):

        tokens = [
            self.id_to_token[token_id]
            for token_id in token_ids
        ]

        return " ".join(
            token
            for token in tokens
            if token not in SPECIAL_TOKENS
        )

    @property
    def vocab_size(self):

        return len(self.token_to_id)
