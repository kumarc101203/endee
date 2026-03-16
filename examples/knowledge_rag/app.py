import os
import streamlit as st
from endee import Endee
from sentence_transformers import SentenceTransformer
from openai import OpenAI

INDEX = "knowledge_base"

st.set_page_config(page_title="Endee AI Assistant", layout="wide")

st.title("Endee Documentation AI Assistant")

# ---------- Load resources only once ----------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_endee_index():
    client = Endee()
    return client.get_index(INDEX)

@st.cache_resource
def load_llm():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error("OPENAI_API_KEY environment variable not set.")
        st.stop()

    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

model = load_embedding_model()
index = load_endee_index()
llm = load_llm()

# ---------- Chat memory ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
query = st.chat_input("Ask a question about Endee")

if query:

    # show user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        with st.spinner("Searching knowledge base..."):

            # Generate embedding
            query_embedding = model.encode(query).tolist()

            # Vector search
            results = index.query(
                vector=query_embedding,
                top_k=3
            )

            context = [r["meta"]["text"] for r in results]
            context_text = "\n".join(context)

            prompt = f"""
You are an assistant answering questions about Endee.

Context:
{context_text}

Question:
{query}

Answer clearly using the context.
"""

            response = llm.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )

            answer = response.choices[0].message.content

        st.markdown(answer)

        # show retrieved context
        with st.expander("Retrieved Context"):
            for c in context:
                st.write("-", c)

    st.session_state.messages.append({"role": "assistant", "content": answer})