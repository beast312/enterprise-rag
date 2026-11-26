"""Answer questions using the FAISS index plus an LLM (Groq by default, OpenAI as fallback)."""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.embeddings import get_embeddings

load_dotenv()
INDEX_DIR = Path(__file__).resolve().parent.parent / "vectorstore"

PROMPT = ChatPromptTemplate.from_template(
    "Answer the question using only the context below. If the answer is not in the context, say so.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


def _real(key):
    value = os.getenv(key, "")
    return value if value and not value.startswith("your-") else ""


def get_llm():
    if _real("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        return ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"), temperature=0)
    if _real("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    raise RuntimeError("Set GROQ_API_KEY (free at console.groq.com/keys) in .env")


def build_chain():
    store = FAISS.load_local(str(INDEX_DIR), get_embeddings(), allow_dangerous_deserialization=True)
    retriever = store.as_retriever(search_kwargs={"k": 4})
    fmt = lambda docs: "\n\n".join(d.page_content for d in docs)
    return {"context": retriever | fmt, "question": RunnablePassthrough()} | PROMPT | get_llm() | StrOutputParser()


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    chain = build_chain()
    while (q := input("Ask (blank to quit): ").strip()):
        print(chain.invoke(q), "\n")
