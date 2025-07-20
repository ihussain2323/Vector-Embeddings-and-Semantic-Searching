# Vector Embeddings & Semantic PDF Search

A Python project to index and semantically search a PDF using vector embeddings, Pinecone, and LangChain. This project demonstrates how to turn a PDF into a searchable knowledge base using state-of-the-art NLP techniques.

---

## Features
- **PDF Indexing:** Splits a PDF into overlapping text chunks and generates embeddings for each chunk.
- **Vector Database:** Stores embeddings and metadata in a Pinecone index for fast similarity search.
- **Semantic Search:** Query the PDF using natural language and retrieve the most relevant passages.
- **Interactive CLI:** Simple command-line interface for searching the indexed PDF.

---

## Directory Structure
```
Assignment 1/
├── Alexander_the_Great.pdf     # PDF to index and search
├── PDF_Indexer.py              # Script to build/update Pinecone index
├── Search.py                   # CLI for semantic search
├── requirements.txt            # Python dependencies
├── .gitignore                  # Files/folders to ignore in git
└── README.md                   # Project documentation
```

---

## Setup Instructions

### 1. Prerequisites
- Python 3.9 or 3.10
- Pinecone account ([sign up free](https://www.pinecone.io))
- pip (Python package manager)

### 2. Clone the Repository
```sh
git clone https://github.com/ihussain2323/Vector-Embeddings-and-Semantic-Searching.git
cd Vector-Embeddings-and-Semantic-Searching
```

### 3. Create a Virtual Environment
```sh
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows
```

### 4. Install Dependencies
```sh
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Set Up Environment Variables
Create a `.env` file in the project root with your Pinecone credentials:
```
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENV=your-pinecone-environment
```

---

## Usage

### 1. Index the PDF
Run the indexer script to process the PDF and upload embeddings to Pinecone:
```sh
python PDF_Indexer.py
```
**What happens:**
- Loads and splits the PDF into 1,000-character chunks (200 overlap)
- Embeds each chunk using the 'all-MiniLM-L6-v2' model
- Creates (if needed) a Pinecone index named `pdf-index`
- Upserts all vectors and metadata

### 2. Search the PDF
Run the search script and enter your questions interactively:
```sh
python Search.py
```
- Enter a natural language question about Alexander the Great.
- The script returns the top 3 most relevant passages (with score, page, and snippet).

---

## Example Output
```
Enter a question: What battles did Alexander fight?

Top 3 results for “What battles did Alexander fight?”:
(0.8123) p. 12: ...description of the Battle of Gaugamela and its significance...
(0.7991) p. 7: ...account of the Siege of Tyre and military tactics...
(0.7884) p. 15: ...details on the Battle of Issus and aftermath...
```

---

## Environment Variables
- `PINECONE_API_KEY` – Your Pinecone API key
- `PINECONE_ENV` – Your Pinecone environment (e.g., us-east-1-gcp)

---

## License
This project is for educational purposes.

---

## Credits
- [LangChain](https://github.com/langchain-ai/langchain)
- [Pinecone](https://www.pinecone.io)
- [Sentence Transformers](https://www.sbert.net/)

---

Feel free to contribute or open issues for improvements!
