import os
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

# Groq client
client = Groq(api_key=GROQ_API_KEY)


def query_llm_with_context(query: str, context: str):

    system_prompt = """
You are a helpful university assistant.

Answer the user's question ONLY using the provided context.

If the context does not contain the answer, say:
"I cannot answer this based on the available documents."
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


def ask_question(question):

    # convert query to embedding
    query_vector = model.encode(question).tolist()

    # search pinecone
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
    print("\nSources:")
    for s in set(sources):
        print("-", s)
    if context == "":
        print("No relevant information found.")
        return

    answer = query_llm_with_context(question, context)

    print("\nAnswer:\n")
    print(answer)
    print("\n----------------------\n")


print("\n🎓 University Assistant Ready")
print("Type 'exit' to quit\n")

while True:

    q = input("Ask: ")

    if q.lower() == "exit":
        break

    ask_question(q)