Plooral Assignment 1: PDF Indexer & Search

A self-contained project demonstrating how to:

Index a PDF (Alexander_the_Great.pdf) into Pinecone using LangChain embeddings.

Perform interactive semantic search against that index.

Project Structure

Assignment 1/                   # Root folder
├── Alexander_the_Great.pdf     # PDF to index
├── PDF_Indexer.py              # Builds/updates Pinecone index
├── Search.py                   # CLI for semantic search
├── requirements.txt            # Pinned Python dependencies
└── README.md                   # This documentation

Ignored: virtual environment folders, IDE settings, OS‑specific files.

Prerequisites

Python 3.9 or 3.10

Pinecone account (free tier)

Get an API key + environment at https://www.pinecone.io

pip (Python package manager)

Setup (no Git required)

Obtain the project folder.

If sent by email or cloud drive, download and unzip it.

Or copy it via Finder/File Explorer or command line, e.g.:

cp -R "/path/to/downloads/Assignment 1" ~/projects/

Change into the project directory:

cd "~/projects/Assignment 1"

Create a .env file alongside PDF_Indexer.py with:

PINECONE_API_KEY=<YOUR_API_KEY>
PINECONE_ENV=<YOUR_ENVIRONMENT>

Provision a virtual environment and activate it:

python3 -m venv .venv
source .venv/bin/activate     # macOS/Linux
.\.venv\Scripts\activate    # Windows

Install dependencies:

pip install --upgrade pip
pip install -r requirements.txt

Usage

1. Build the Pinecone Index

python PDF_Indexer.py

What happens:

Loads & splits the PDF into 1,000‑character chunks (200 overlap)

Embeds each chunk via all-MiniLM-L6-v2

Creates (if needed) a Pinecone index named pdf-index

Upserts all vectors + metadata

Sample output:

Loading PDF: Alexander_the_Great.pdf
Loading complete - 43 pages loaded
Split into 228 chunks
Creating embeddings using 'all-MiniLM-L6-v2' model…
Upserting vectors into Pinecone index…
Success! PDF is searchable in Pinecone.

2. Launch Interactive Search

python Search.py

Enter your queries about Alexander the Great.

The script returns the top 3 matching chunks (score, page, snippet).
