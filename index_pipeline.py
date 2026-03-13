import os
import re
from pypdf import PdfReader
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# connect pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "university-rag"

# create index if not exists
existing_indexes = [i.name for i in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)


def clean_text(text):

    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\.{2,}', '', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def extract_text(pdf_path):

    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    return clean_text(text)


def chunk_text(text, size=300, overlap=50):

    chunks = []

    for i in range(0, len(text), size - overlap):
        chunk = text[i:i+size]

        if len(chunk) > 100:
            chunks.append(chunk)

    return chunks


DATA_FOLDER = "data"

vector_id = 0

for file in os.listdir(DATA_FOLDER):

    if file.endswith(".pdf"):

        print("Processing", file)

        text = extract_text(os.path.join(DATA_FOLDER, file))

        chunks = chunk_text(text)

        for chunk in chunks:

            vector = model.encode(chunk).tolist()

            index.upsert(
                vectors=[
                    {
                        "id": str(vector_id),
                        "values": vector,
                        "metadata": {
                            "text": chunk,
                            "source": file
                        }
                    }
                ]
            )

            vector_id += 1

print("✅ Documents embedded successfully!")