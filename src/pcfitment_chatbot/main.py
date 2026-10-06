from dotenv import load_dotenv

load_dotenv()

from rag import sync_vector_database


def main():

    print("=" * 60)
    print("            PCFitment Support Chatbot")
    print("=" * 60)

    # Must run BEFORE importing graph so every article DB exists first.
    print("\nChecking vector databases...")
    sync_vector_database()

    from graph import build_graph

    print("\nLoading LangGraph...")
    graph = build_graph()

    print("\n" + "=" * 60)
    print("Chatbot is ready!")
    print("Ask questions about PCFitment.")
    print("Type 'exit' to close.")
    print("=" * 60)

    history = []

    while True:

        question = input("\nYou: ").strip()

        if not question:
            continue

        if question.lower() in ["exit", "quit", "q"]:
            print("\nGoodbye!")
            break

        try:
            result = graph.invoke(
                {
                    "question": question,
                    "history": history,
                    "standalone": "",
                    "context": "",
                    "answer": "",
                    "relevant": False,
                    "sources": [],
                }
            )

            print("\nBot:")
            print(result["answer"])

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": result["answer"]})
    
            sources = result.get("sources", [])
            if sources:
                print("\nRead more:")
                for i, source in enumerate(sources, start=1):
                    print(f"  {i}. {source['title']}")
                    print(f"     {source['url']}")

        except Exception as error:
            print("\nError while processing question:")
            print(error)


if __name__ == "__main__":
    main()