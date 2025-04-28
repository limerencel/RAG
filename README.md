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

2. Install dependencies and the CLI tool:

   ```
   # Install in development mode (recommended)
   pip install -e .

   # Or install as a regular package
   pip install .
   ```

3. Create a `.env` file in your working directory with your xAI API key:
   ```
   XAI_API_KEY="your-api-key-here"
   ```

## Usage

### CLI Command

After installation, you can use the `ragchat` command directly from your terminal:

```bash
# Process all documents in a directory
ragchat --dir /path/to/your/docs

# Process specific files
ragchat --files document1.txt document2.pdf notes.md

# Specify where to store the vector database
ragchat --dir /path/to/docs --persist ./my_vector_db

# Specify conversation history file
ragchat --dir /path/to/docs --history my_chat_history.pkl
```

Available options:

- `--dir`, `-d`: Directory containing documents to process
- `--files`, `-f`: Specific files to process (space-separated)
- `--persist`, `-p`: Directory to persist vector database (default: ./chroma_db)
- `--history`: File to save conversation history (default: rag_conversation_history.pkl)

### Python Module Usage

You can also use the script directly if you prefer:

```bash
# Run the application with default settings
python -m ragchat.cli
```

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

## Important Note on File Handling and Vector Storage

### How Files Are Processed

When you add files to the RAG system, they are:

1. Split into text chunks
2. Converted to vector embeddings
3. Stored in a Chroma vector database (SQLite file)

### Vector Database Behavior

- **Additive storage**: By default, new documents are added to the existing vector database when using the same `--persist` directory
- **No automatic differentiation**: The system doesn't automatically distinguish between documents from different sessions
- **Storage location**: Vectors are stored in `./chroma_db/` by default (or your custom location with `--persist`)

### Avoiding Data Pollution

To prevent mixing document sets or to manage different knowledge bases:

1. **Use separate vector databases for different document sets**:

   ```bash
   # Financial documents in one database
   ragchat --dir financial/ --persist ./financial_vectors

   # Legal documents in a different database
   ragchat --dir legal/ --persist ./legal_vectors
   ```

2. **Clear existing vectors before adding new documents**:

   - Delete the vector database directory (e.g., `./chroma_db/`) before processing a new set of documents
   - Or use a new directory with the `--persist` flag

3. **For single-session use cases**:
   - Use an ephemeral database by omitting the `--persist` flag (vectors will be stored in memory only)

The vector database holds all information the LLM will use during retrieval - the original documents are not accessed again after processing.

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
