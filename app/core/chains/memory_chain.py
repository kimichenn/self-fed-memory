from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

MEMORY_TEMPLATE = """You are a helpful AI assistant.
Your task is to determine if the user wants to save a memory.
If so, extract the memory and respond with a confirmation.
If not, respond with a generic message.

User query: {query}
"""


def get_memory_chain(llm: BaseChatModel = None) -> Runnable:
    """Create a chain that determines if a user wants to save a memory."""
    llm = llm or ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0)
    prompt = ChatPromptTemplate.from_template(MEMORY_TEMPLATE)
    return prompt | llm | StrOutputParser()
