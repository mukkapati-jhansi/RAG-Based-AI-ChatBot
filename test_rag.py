from app.graph import rag_graph


question = "What is Agentic AI?"


result = rag_graph.invoke(
    {
        "question": question,
        "context": [],
        "answer": "",
        "confidence": 0.0,
    }
)


print("\n" + "=" * 60)
print("QUESTION")
print("=" * 60)
print(question)

print("\n" + "=" * 60)
print("ANSWER")
print("=" * 60)
print(result["answer"])

print("\n" + "=" * 60)
print("RETRIEVAL SCORE")
print("=" * 60)
print(result["confidence"])

print("\n" + "=" * 60)
print("RETRIEVED CONTEXT")
print("=" * 60)

for i, item in enumerate(result["context"], start=1):
    print(f"\n--- Context {i} ---")
    print(f"Page: {item['page']}")
    print(f"Score: {item['score']}")
    print(f"Source: {item['source']}")
    print(item["text"][:500])