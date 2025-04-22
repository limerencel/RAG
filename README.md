# Financial Document RAG

A Retrieval-Augmented Generation (RAG) system for querying information from financial documents, specifically SEC 10-K filings.

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

1. Update the `directory` variable in `main.py` to point to your document directory:

   ```python
   directory = r"path/to/your/documents"  # Directory with financial documents
   ```

2. Run the application:

   ```
   python main.py
   ```

3. The system will:

   - Load documents from the specified directory
   - Process and embed them
   - Save the vector database to `./chroma_db/`
   - Launch an interactive chat interface

4. Available commands in the chat interface:
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
