# PDF_Indexer.py
# This script loads a PDF, splits it into chunks, generates embeddings,
# and upserts them into a Pinecone serverless index.

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


def main():
    """Main function that handles PDF processing and upsert to Pinecone"""

    # 0. Load environment vars and init Pinecone client
    load_dotenv()  # expects PINECONE_API_KEY & PINECONE_ENV in .env
    api_key = os.getenv("PINECONE_API_KEY")
    env = os.getenv("PINECONE_ENV")
    if not api_key or not env:
        raise ValueError("PINECONE_API_KEY and PINECONE_ENV must be set in environment or .env file.")

    pc = Pinecone(api_key=api_key, environment=env)
    index_name = "pdf-index"

    # 1. Create/check the index
    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    dim = len(embed_model.embed_documents([""])[0])

    existing = pc.list_indexes().names()
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    index = pc.Index(index_name)

    # 2. Load PDF and split into chunks
    path_pdf = "Alexander_the_Great.pdf"
    if not os.path.isfile(path_pdf):
        raise FileNotFoundError(f"PDF not found at {path_pdf}")

    print(f"Loading PDF: {path_pdf}")
    loader = PyPDFLoader(path_pdf)
    pages = loader.load()
    print(f"Loading complete - {len(pages)} pages loaded")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks")

    # 3. Create embeddings
    print("Creating embeddings using 'all-MiniLM-L6-v2' model…")
    texts = [chunk.page_content for chunk in chunks]
    metas = [chunk.metadata for chunk in chunks]
    vectors = embed_model.embed_documents(texts)

    # 4. Prepare and upsert
    print("Upserting vectors into Pinecone index…")
    upsert_data = []
    for i, (vec, meta, chunk) in enumerate(zip(vectors, metas, chunks)):
        meta["text"] = chunk.page_content
        upsert_data.append((str(i), vec, meta))

    index.upsert(vectors=upsert_data)
    print("Success! PDF is searchable in Pinecone.")


if __name__ == "__main__":
    main()