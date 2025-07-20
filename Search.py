# Search.py
import os
import re
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def main():
    # Load environment and initialize Pinecone
    load_dotenv()  # expects PINECONE_API_KEY & PINECONE_ENV in .env
    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENV")
    )
    index = pc.Index("pdf-index")

    # Load & split the PDF locally into the same chunks you indexed
    pdf_path = "Alexander_the_Great.pdf"
    loader   = PyPDFLoader(pdf_path)
    pages    = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks   = splitter.split_documents(pages)

    # Prompt the user
    q = input("Enter a question: ").strip()
    if not q:
        print("No question given.")
        return

    # Embed the query and do a vector search
    embedder     = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    query_vector = embedder.embed_query(q)
    resp         = index.query(vector=query_vector, top_k=3, include_metadata=True)

    # Extract meaningful keywords from the query
    tokens    = re.findall(r"\w+", q.lower())
    stopset   = {"the","and","for","with","that","this","is","of","what","when","where","which","who","whom","why","how","alexander"}
    keywords  = [t for t in tokens if len(t)>2 and t not in stopset]

    # Check top-3 matches for any keyword
    found = False
    print(f"\nTop 3 results for “{q}”:\n")
    for match in resp.matches:
        idx  = int(match.id)
        text = chunks[idx].page_content
        if any(kw in text.lower() for kw in keywords):
            page    = chunks[idx].metadata.get("page", "n/a")
            snippet = text.replace("\n"," ")[:200]
            if len(text)>200: snippet += "…"
            print(f"({match.score:.4f}) p. {page}: {snippet}\n")
            found = True

    # Fallback: plain substring scan across all chunks
    if not found:
        print("No keyword in top 3 — falling back to substring search:\n")
        for chunk in chunks:
            content_lower = chunk.page_content.lower()
            if any(kw in content_lower for kw in keywords):
                page    = chunk.metadata.get("page", "n/a")
                snippet = chunk.page_content.replace("\n"," ")[:200]
                if len(chunk.page_content)>200: snippet += "…"
                print(f"(0.0000) p. {page}: {snippet}\n")
                break

if __name__ == "__main__":
    main()
