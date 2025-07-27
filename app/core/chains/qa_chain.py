from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

QA_TEMPLATE = """You are a personal AI assistant for {name}.
You have access to the user's personal notes and memories.
Use the following information to answer the question.

{context}

User's question: {question}
Answer in a helpful manner based on the above context and the user's preferences.
"""


def get_qa_chain(llm: BaseChatModel = None) -> Runnable:
    """Build a chain that answers questions based on context."""
    llm = llm or ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0)
    prompt = ChatPromptTemplate.from_template(QA_TEMPLATE)
    return prompt | llm | StrOutputParser()
