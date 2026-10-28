from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.mistral_client import create_llm, invoke_with_retry

def get_llm(max_tokens=1024):
    return create_llm(temperature=0.3, max_tokens=max_tokens)


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 3000,
        chunk_overlap = 200
    )

    return splitter.split_text(transcript)

def summarize(transcript : str) -> str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", "Summarize this meeting transcript concisely in professional bullet points."),
        ("human", "{text}"),
    ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    if not chunks:
        return "No transcript content to summarize."

    chunk_summaries = [invoke_with_retry(map_chain, {"text" : chunk}) for chunk in chunks]

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
        (
            "system",
            "You are an expert meeting summarizer. Combine these partial summaries "
            "into one final professional meeting summary in bullet points.",
        ),
        ("human", "{text}"),
    ]
    )

    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | combined_prompt | llm | StrOutputParser()
    )

    return invoke_with_retry(combined_chain, combined)

def generate_title(transcipt : str) -> str:
    llm = get_llm(max_tokens=64)

    

    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | 
        ChatPromptTemplate.from_messages([
             (
                "system",
                "Based on the meeting transcript, generate a short professional meeting title "
                "(max 8 words). Only return the title, nothing else.",
            ),
            ("human", "{text}"),
        ])
        | llm
        |StrOutputParser()
    )

    return invoke_with_retry(title_chain, transcipt[:2000])



