import re
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from config import OLLAMA_BASE_URL

llm = Ollama(model="deepseek-r1:1.5b", base_url=OLLAMA_BASE_URL)

def generate_answer(user_query, context_documents, chat_history=None):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    history_text = "\n".join(
    [f"{msg['role']}: {msg['content']}" for msg in (chat_history or []) if msg.get("content") is not None]
)

    prompt_template = """
You are a helpful assistant that answers **only based on the provided context** below.
- If the answer is not present in the context, reply with: "I don’t know based on the provided document."
- answer the question in clear paragraphs. Use bullet points or numbered lists if necessary.
- Do not use prior knowledge.

Context:
{document_context}

Chat History:
{chat_history}

User Query:
{user_query}

Answer:
"""


    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    result = chain.invoke({"user_query": user_query, "document_context": context_text, "chat_history": history_text})
    return re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()