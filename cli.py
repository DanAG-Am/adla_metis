def main():
    print("BusinessBrain")
    print("Local business analytics assistant")
    print()
    print("Type 'exit' to quit.")
    print()

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        print()
        print("BusinessBrain:")
        print(f"You asked: {question}")
        print()
        print("The LLM is not connected yet.")
        print()

if __name__ == "__main__":
    main()
