# General-Purpose Document RAG

A Retrieval-Augmented Generation (RAG) system for querying information from any collection of documents.

## Setup

### Prerequisites

- Python 3.12 or newer
- An xAI API key for access to Grok-3-Beta

### Installation

1. Clone this repository:

   ```
   git clone <repository-url>
   cd rag
   ```

2. Install dependencies using your preferred method:

   **Using pip**:

   ```
   pip install -r requirements.txt
   ```

   **Using uv**:

   ```
   uv pip install -e .
   ```

3. Create a `.env` file in the project root with your xAI API key:
   ```
   XAI_API_KEY="your-api-key-here"
   ```

## Usage

### Basic Usage

Run the application with default settings:

```
python main.py
```

### Command Line Interface (CLI)

The application supports command-line arguments for specifying documents to process:

```
# Process all documents in a directory
python main.py --dir /path/to/your/docs

# Process specific files
python main.py --files document1.txt document2.pdf notes.md

# Specify where to store the vector database
python main.py --dir /path/to/docs --persist ./my_vector_db

# Specify conversation history file
python main.py --dir /path/to/docs --history my_chat_history.pkl
```

Available options:

- `--dir`, `-d`: Directory containing documents to process
- `--files`, `-f`: Specific files to process (space-separated)
- `--persist`, `-p`: Directory to persist vector database (default: ./chroma_db)
- `--history`: File to save conversation history (default: rag_conversation_history.pkl)

### Interactive Chat

The system will:

- Load documents from the specified sources
- Process and embed them
- Save the vector database to the specified persistence directory
- Launch an interactive chat interface

Available commands in the chat interface:

- Ask any question about the documents
- Type `history` to view conversation history
- Type `clear` to clear conversation history
- Type `exit` to quit the application

## How It Works

This RAG application follows these steps:

1. **Document Loading**: Loads text files from a specified directory
2. **Text Chunking**: Divides documents into manageable chunks
3. **Embedding Generation**: Creates vector embeddings using sentence-transformers
4. **Vector Storage**: Stores embeddings in a Chroma vector database
5. **Querying**: Accepts user questions, finds relevant document chunks
6. **Response Generation**: Uses Grok-3-Beta LLM to generate answers based on retrieved context

## Customization

- Change the chunk size in `load_documents()` function
- Modify the prompt template in `create_rag_chain()` function
- Adjust the number of retrieved documents with `search_kwargs={"k": 4}`

## Persistence

- The vector database is persisted to `./chroma_db/`
- Conversation history is saved to `rag_conversation_history.pkl`

## Supported Document Types

The system supports various file formats:

- Text files (.txt)
- PDFs (using PyPDFLoader)
- Other document types (using UnstructuredFileLoader)

You can extend the system to support additional file types by adding appropriate document loaders.
