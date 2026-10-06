import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, convert_to_messages
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")
RETRIEVAL_K = 10

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

LOCAL_DEMO_KNOWLEDGE = {
    "insurellm": "Insurellm is a sample company context used for demo conversations in this project.",
    "rag": "RAG combines retrieval with generation to ground answers in relevant documents.",
    "llm": "LLMs are large language models trained on broad text corpora for language understanding and reasoning.",
    "agent": "Agents can reason over tasks, use tools, and coordinate workflows with external systems.",
}


def _build_llm_components():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, None, None

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(temperature=0, model_name=MODEL)
    return retriever, llm, embeddings


retriever, llm, _ = _build_llm_components()


def fetch_context(question: str) -> list[Document]:
    """Retrieve relevant context documents for a question."""
    if retriever is None:
        lowered = question.lower()
        matched = [
            Document(page_content=content, metadata={"source": "offline-demo"})
            for keyword, content in LOCAL_DEMO_KNOWLEDGE.items()
            if keyword in lowered
        ]
        if not matched:
            return [
                Document(
                    page_content="This is a local demo mode. Configure OPENAI_API_KEY to enable full retrieval-based answers.",
                    metadata={"source": "offline-demo"},
                )
            ]
        return matched
    return retriever.invoke(question, k=RETRIEVAL_K)


def combined_question(question: str, history: list[dict] = []) -> str:
    """Combine all the user's messages into a single string."""
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """Answer the given question with RAG when an API key is available; otherwise use demo mode."""
    combined = combined_question(question, history)
    docs = fetch_context(combined)

    if llm is None:
        context_text = "\n\n".join(doc.page_content for doc in docs)
        response = (
            "This is the local demo mode for the LLM engineering project. "
            "No OPENAI_API_KEY is configured, so the app is running with a lightweight fallback answer. "
            f"Context summary: {context_text}"
        )
        return response, docs

    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs
