from langgraph.graph import StateGraph, START, END

from app.state import RAGState
from app.retriever import retrieve_documents
from app.llm import llm
from app.prompts import build_prompt


# ---------------------------------------------------------
# Node 1: Retrieve relevant documents
# ---------------------------------------------------------

def retrieve_node(state: RAGState):

    question = state["question"]

    contexts, confidence = retrieve_documents(
        question=question,
        top_k=5,
    )

    return {
        "context": contexts,
        "confidence": confidence,
    }


# ---------------------------------------------------------
# Node 2: Generate grounded answer
# ---------------------------------------------------------

def generate_node(state: RAGState):

    question = state["question"]
    contexts = state["context"]
    confidence = state["confidence"]

    # Do not generate an answer when retrieval is too weak.
    if not contexts or confidence < 0.60:
        return {
            "answer": (
                "I couldn't find enough information about this "
                "in the provided Agentic AI eBook."
            )
        }

    context_text = "\n\n".join(
        [
            (
                f"[Page {item['page']} | "
                f"Similarity: {item['score']}]\n"
                f"{item['text']}"
            )
            for item in contexts
        ]
    )

    prompt = build_prompt(
        question=question,
        context=context_text,
    )

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
    }

# ---------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------

graph_builder = StateGraph(RAGState)

graph_builder.add_node(
    "retrieve",
    retrieve_node,
)

graph_builder.add_node(
    "generate",
    generate_node,
)

graph_builder.add_edge(
    START,
    "retrieve",
)

graph_builder.add_edge(
    "retrieve",
    "generate",
)

graph_builder.add_edge(
    "generate",
    END,
)

rag_graph = graph_builder.compile()