#Actionableitems , decision , questions 

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, ConfigDict, Field
from core.mistral_client import MistralAPIError, create_llm, invoke_with_retry


def get_llm():
    return create_llm(temperature=0.2, max_tokens=2048)


class MeetingDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action_items: str = Field(description="Numbered tasks with owner and deadline; use Not specified for missing details.")
    key_decisions: str = Field(description="Numbered list of key decisions.")
    open_questions: str = Field(description="Numbered list of unresolved questions or follow-up topics.")


def extract_meeting_details(transcript: str) -> dict:
    """Extract all three sections in one request without dropping transcript text."""
    parser = PydanticOutputParser(pydantic_object=MeetingDetails)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert meeting analyst. Extract action items (task, owner, deadline), "
         "key decisions, and unresolved questions from the transcript. Use numbered lists "
         "inside each JSON string. If a section is empty, use 'No action items found.', "
         "'No key decisions found.', or 'No open questions found.' respectively. "
         "Do not invent facts. Return only JSON.\n{format_instructions}"),
        ("human", "{text}"),
    ]).partial(format_instructions=parser.get_format_instructions())
    chain = prompt | get_llm().bind(response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "meeting_details",
            "schema": MeetingDetails.model_json_schema(),
            "strict": True,
        },
    })
    response = invoke_with_retry(chain, {"text": transcript})
    try:
        return parser.invoke(response).model_dump()
    except OutputParserException:
        raise MistralAPIError(
            "Mistral returned incomplete or invalid meeting details. "
            "No automatic retry was made to avoid extra API usage."
        ) from None



def build_chain(system_prompt : str):
    llm = get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) |ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human","{text}"),
    ]) | llm |StrOutputParser()
    )

def extract_action_items(transcript:str)->str:
    chain = build_chain(
         "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )

    return invoke_with_retry(chain, transcript)


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    return invoke_with_retry(chain, transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    return invoke_with_retry(chain, transcript)
