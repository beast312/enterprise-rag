import streamlit as st
from src.rag import build_chain

st.title("Enterprise RAG Q&A")
chain = st.cache_resource(build_chain)()
q = st.text_input("Ask a question about your documents")
if q:
    with st.spinner("Thinking..."):
        st.write(chain.invoke(q))
