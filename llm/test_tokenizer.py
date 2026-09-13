from llm.tokenizer import SimpleTokenizer

def main():

    texts = [
        "What was revenue by region?",
        "How many orders did we have?",
        "Show revenue by product category.",
        "Which customer segment generated the most revenue?",
    ]

    tokenizer = SimpleTokenizer()

    tokenizer.build_vocab(texts)

    print("Vocabulary size:")
    print(tokenizer.vocab_size)

    print()

    text = "What was revenue by region?"

    tokens = tokenizer.tokenize(text)

    print("Tokens:")
    print(tokens)

    print()

    encoded = tokenizer.encode(text)

    print("Encoded:")
    print(encoded)

    print()

    decoded = tokenizer.decode(encoded)

    print("Decoded:")
    print(decoded)

if __name__ == "__main__":
    main()
