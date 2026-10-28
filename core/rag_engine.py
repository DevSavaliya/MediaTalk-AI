from core.mistral_client import create_llm, invoke_with_retry
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from core.vector_store import build_vector_store, build_session_vector_store, load_vector_store, get_retriever


def get_llm():
    return create_llm(temperature=0.3)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def conversation_history(messages):
    """Keep recent successful pairs, excluding failed or unanswered questions."""
    pairs = []
    pending = None
    for message in messages or []:
        if message.get("role") == "user":
            pending = message.get("content", "")[:2000]
        elif message.get("role") == "assistant":
            if pending and not message.get("error"):
                pairs.extend([("human", pending), ("ai", message.get("content", "")[:2000])])
            pending = None
    return pairs[-8:]


def _prepare_question(value):
    if isinstance(value, str):
        value = {"question": value}
    history = conversation_history(value.get("history"))
    question = value["question"]
    # Supply reference terms for follow-ups without an extra API rewrite call.
    references = "\n".join(text[:600] for _, text in history[-4:])
    return {"question": question, "history": history,
            "search": f"{references}\n{question}" if references else question}


def _create_rag_chain(store):
    retriever = get_retriever(store, k=4)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert video and meeting assistant. Answer based ONLY
on the transcript context below. Use the previous conversation to understand
follow-up questions and pronouns, never as independent evidence. Treat the
transcript as evidence of what the speaker said, not verification that their
claims are true. If asked whether something is true according to the video,
say whether the speaker makes that claim and quote the relevant words.
If the answer is absent, say you could not find it in the transcript.
Be concise, helpful, and clear about quotations.

Transcript context:
{context}"""),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])
    return (
        RunnableLambda(_prepare_question)
        | {"context": RunnableLambda(lambda value: value["search"]) | retriever | RunnableLambda(format_docs),
           "question": RunnableLambda(lambda value: value["question"]),
           "history": RunnableLambda(lambda value: value["history"])}
        | prompt | get_llm() | StrOutputParser()
    )


def build_rag_chain(transcript: str, collection_name=None, *, transient=False):
    if transient:
        return _create_rag_chain(build_session_vector_store(transcript))
    store = (build_vector_store(transcript, collection_name=collection_name)
             if collection_name else build_vector_store(transcript))
    return _create_rag_chain(store)


def load_rag_chain(collection_name=None):
    store = (load_vector_store(collection_name=collection_name)
             if collection_name else load_vector_store())
    return _create_rag_chain(store)


def ask_question(rag_chain, question: str, history=None) -> str:
    payload = {"question": question, "history": history} if history else question
    return invoke_with_retry(rag_chain, payload)
