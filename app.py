import os
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import Groq

load_dotenv()

# API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# pinecone connection
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("university-rag")

# groq client
client = Groq(api_key=GROQ_API_KEY)

st.title("🎓 College Assistant")

st.write("Ask questions about university regulations and policies.")


def query_llm(query, context):

    prompt = f"""
You are a helpful university assistant.

Answer the user's question ONLY using the provided context.

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content


def retrieve_context(question):

    query_vector = model.encode(question).tolist()

    results = index.query(
        vector=query_vector,
        top_k=5,
        include_metadata=True
    )

    threshold = 0.55
    context = ""
    sources = []

    for match in results["matches"]:

        if match["score"] >= threshold:
            context += match["metadata"]["text"] + "\n"
            sources.append(match["metadata"]["source"])

    return context[:2000], list(set(sources))


user_question = st.text_input("Ask a question:")

if user_question:

    context, sources = retrieve_context(user_question)

    if context == "":
        st.warning("No relevant information found.")
    else:
        answer = query_llm(user_question, context)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for s in sources:
            st.write("-", s)