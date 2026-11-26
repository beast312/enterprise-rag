"""Answer questions using the FAISS index plus an LLM (Groq if key set, else OpenAI)."""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
INDEX_DIR = Path(__file__).resolve().parent.parent / "vectorstore"

PROMPT = ChatPromptTemplate.from_template(
    "Answer the question using only the context below. If the answer is not in the context, say so.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


def get_llm():
    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def build_chain():
    store = FAISS.load_local(str(INDEX_DIR), OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    retriever = store.as_retriever(search_kwargs={"k": 4})
    fmt = lambda docs: "\n\n".join(d.page_content for d in docs)
    return {"context": retriever | fmt, "question": RunnablePassthrough()} | PROMPT | get_llm() | StrOutputParser()


if __name__ == "__main__":
    chain = build_chain()
    while (q := input("Ask (blank to quit): ").strip()):
        print(chain.invoke(q), "\n")
