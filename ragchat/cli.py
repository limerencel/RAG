import getpass
import os
import argparse
import sys
from typing import List, Dict, Any
import warnings
import time
import pickle
import datetime
warnings.filterwarnings('ignore')

# Load environment variables from .env file
from dotenv import load_dotenv

def setup_api_key():
    """Set up the API key for xAI"""
    print("Loading environment variables from .env file...")
    load_dotenv()

    # Check if API key is in environment
    if os.environ.get("XAI_API_KEY"):
        print("XAI API key found in environment variables")
    else:
        # Fallback to manual input if not in .env
        print("XAI API key not found in environment variables, prompting for input...")
        os.environ["XAI_API_KEY"] = getpass.getpass("Enter API key for xAI: ")

# Import LangChain components
def import_langchain():
    """Import LangChain components"""
    print("Importing LangChain components...")
    global init_chat_model, TextLoader, DirectoryLoader, RecursiveCharacterTextSplitter
    global HuggingFaceEmbeddings, Chroma, create_stuff_documents_chain, create_retrieval_chain
    global ChatPromptTemplate, HumanMessage, SystemMessage, AIMessage, ConversationBufferMemory
    
    from langchain.chat_models import init_chat_model
    from langchain_community.document_loaders import TextLoader, DirectoryLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from langchain.memory import ConversationBufferMemory

# Define a custom display function that works in both Jupyter and standard Python
def safe_display(content):
    try:
        # Try using IPython display if available
        from IPython.display import Markdown, display
        display(Markdown(content))
    except (ImportError, NameError):
        # Fallback for standard Python environment - with some basic formatting
        content = content.replace("**", "\033[1m").replace("**", "\033[0m")  # Bold text
        print(content)

# Initialize LLM
def init_llm():
    """Initialize the LLM model"""
    print("Initializing LLM (this may take a moment)...")
    model = init_chat_model("grok-3-beta", model_provider="xai")
    print("LLM initialized successfully")
    return model

# Function to load and process documents
def load_documents(file_paths=None, directory_path=None):
    print("Loading documents...")
    documents = []
    
    if file_paths:
        from langchain_community.document_loaders import TextLoader, UnstructuredEPubLoader, UnstructuredFileLoader, PyPDFLoader
        for file in file_paths:
            print(f"Loading file: {file}")
            try:
                ext = os.path.splitext(file)[1].lower()
                if ext == '.txt':
                    loader = TextLoader(file)
                elif ext == '.pdf':
                    loader = PyPDFLoader(file)
                elif ext == '.epub':
                    loader = UnstructuredEPubLoader(file)
                else:
                    loader = UnstructuredFileLoader(file)
                documents.extend(loader.load())
                print(f"  - Loaded file: {file}")
            except Exception as e:
                print(f"Error loading file {file}: {str(e)}")
    
    if directory_path:
        print(f"Loading documents from directory: {directory_path}")
        try:
            # Manually find and load all .txt files
            import glob
            txt_files = glob.glob(os.path.join(directory_path, "**/*.txt"), recursive=True)
            print(f"Found {len(txt_files)} .txt files in directory")
            
            if len(txt_files) == 0:
                print("WARNING: No .txt files found in the directory. Checking for other file types...")
                all_files = glob.glob(os.path.join(directory_path, "**/*.*"), recursive=True)
                file_extensions = set([os.path.splitext(f)[1] for f in all_files])
                print(f"Found files with extensions: {file_extensions}")
                
                # Try loading with a different extension if no txt files found
                if len(all_files) > 0:
                    print("Trying to load files with other extensions...")
                    for file in all_files[:5]:  # Try first 5 files
                        try:
                            ext = os.path.splitext(file)[1].lower()
                            print(f"Trying to load {file}")
                            if ext == '.pdf':
                                loader = PyPDFLoader(file)
                                documents.extend(loader.load())
                            elif ext == '.epub':
                                loader = UnstructuredEPubLoader(file)
                                documents.extend(loader.load())
                            else:
                                loader = UnstructuredFileLoader(file)
                                documents.extend(loader.load())
                            print(f"Successfully loaded {file}")
                        except Exception as e:
                            print(f"Error loading {file}: {str(e)}")
            else:
                # Load each txt file individually
                for file_path in txt_files:
                    try:
                        print(f"Loading file: {file_path}")
                        loader = TextLoader(file_path)
                        docs = loader.load()
                        documents.extend(docs)
                        print(f"  - Loaded {len(docs)} documents from {file_path}")
                    except Exception as e:
                        print(f"Error loading {file_path}: {str(e)}")
                
        except Exception as e:
            print(f"Error loading directory {directory_path}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    if not documents:
        raise ValueError("No documents loaded. Please provide valid file paths or directory.")
    
    # Split documents into chunks
    print(f"Splitting {len(documents)} documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    
    print(f"Loaded {len(documents)} documents and split into {len(splits)} chunks.")
    return splits

# Function to set up vector store
def setup_vectorstore(documents, persist_directory=None):
    print("Setting up vector store with embeddings (this may take a while)...")
    start_time = time.time()
    
    # Use HuggingFace embeddings (no API key needed)
    print("Initializing embedding model...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("Embedding model initialized successfully")
    except Exception as e:
        print(f"Error initializing embedding model: {str(e)}")
        raise
    
    # Create and persist vector store if directory is provided
    if persist_directory:
        print(f"Creating vector store and persisting to {persist_directory}...")
        try:
            vectorstore = Chroma.from_documents(
                documents=documents, 
                embedding=embeddings,
                persist_directory=persist_directory
            )
            vectorstore.persist()
            print(f"Vector store created and persisted to {persist_directory}")
        except Exception as e:
            print(f"Error creating persistent vector store: {str(e)}")
            raise
    else:
        print("Creating ephemeral vector store...")
        try:
            vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)
            print("Ephemeral vector store created (not persisted)")
        except Exception as e:
            print(f"Error creating ephemeral vector store: {str(e)}")
            raise
    
    elapsed_time = time.time() - start_time
    print(f"Vector store setup completed in {elapsed_time:.2f} seconds")
    return vectorstore

# Function to create RAG chain
def create_rag_chain(vectorstore, model):
    print("Setting up retriever...")
    try:
        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        print("Retriever set up successfully")
    except Exception as e:
        print(f"Error setting up retriever: {str(e)}")
        raise
    
    # Create prompt template
    print("Creating prompt template...")
    general_docs_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based on the provided documents.
        Answer the question based ONLY on the following context from the documents. If you don't know or can't find the answer in the context, say so.
        Be precise and cite specific information from the documents when possible.
        
        Context:
        {context}"""),
        ("human", "{input}")
    ])
    
    # Create document chain
    print("Creating document chain...")
    try:
        document_chain = create_stuff_documents_chain(model, general_docs_prompt)
        print("Document chain created successfully")
    except Exception as e:
        print(f"Error creating document chain: {str(e)}")
        raise
    
    # Create retrieval chain
    print("Creating retrieval chain...")
    try:
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        print("Retrieval chain created successfully")
    except Exception as e:
        print(f"Error creating retrieval chain: {str(e)}")
        raise
    
    return retrieval_chain

# Function to save conversation history
def save_conversation(conversation_history, filename="conversation_history.pkl"):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(conversation_history, f)
        print(f"Conversation saved to {filename}")
    except Exception as e:
        print(f"Error saving conversation history: {str(e)}")
        # Don't raise, just log the error

# Function to load conversation history
def load_conversation(filename="conversation_history.pkl"):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"No saved conversation found at {filename}, starting new conversation.")
        return []

# Interactive RAG Chat
class RAGChat:
    def __init__(self, rag_chain, load_history=True, history_file="conversation_history.pkl"):
        print("Initializing RAGChat class...")
        self.rag_chain = rag_chain
        self.history_file = history_file
        
        try:
            if load_history:
                print(f"Attempting to load conversation history from {history_file}")
                self.conversation_history = load_conversation(history_file)
            else:
                print("Starting with empty conversation history")
                self.conversation_history = []
            print(f"Conversation history has {len(self.conversation_history)} messages")
        except Exception as e:
            print(f"Error loading conversation history: {str(e)}")
            self.conversation_history = []
        
    def chat(self, query):
        print(f"Processing query: '{query}'")
        try:
            # Process query through RAG chain
            print("Sending query to RAG chain...")
            response = self.rag_chain.invoke({"input": query})
            print("Received response from RAG chain")
            
            # Update conversation history
            print("Updating conversation history...")
            self.conversation_history.append({"role": "human", "content": query, "timestamp": datetime.datetime.now()})
            self.conversation_history.append({"role": "assistant", "content": response["answer"], "timestamp": datetime.datetime.now()})
            
            # Save updated history
            print("Saving conversation history...")
            save_conversation(self.conversation_history, self.history_file)
            
            # Print the response directly for terminal users
            print("\n\033[1mAssistant:\033[0m", response["answer"])
            
            # Return response
            return response["answer"]
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return f"Sorry, I encountered an error: {error_msg}"
    
    def display_history(self):
        try:
            if not self.conversation_history:
                print("No conversation history to display")
                return
                
            print(f"Displaying {len(self.conversation_history)} messages from history")
            for message in self.conversation_history:
                role = message["role"]
                content = message["content"]
                timestamp = message["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                
                if role == "human":
                    safe_display(f"**User ({timestamp}):** {content}")
                else:
                    safe_display(f"**Assistant ({timestamp}):** {content}")
        except Exception as e:
            print(f"Error displaying conversation history: {str(e)}")

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='RAG Chat CLI - Chat with your documents using Retrieval-Augmented Generation')
    parser.add_argument('--dir', '-d', type=str, help='Directory containing documents to process')
    parser.add_argument('--files', '-f', nargs='+', help='Specific files to process (space-separated)')
    parser.add_argument('--persist', '-p', type=str, default='./chroma_db', help='Directory to persist vector database')
    parser.add_argument('--history', type=str, default='rag_conversation_history.pkl', help='File to save conversation history')
    
    # Parse arguments
    args = parser.parse_args()
    
    print("\n=== Starting RAG Application ===\n")
    
    # Set up paths based on arguments or defaults
    directory = args.dir if args.dir else r"./test_docs"  # Directory with documents (default to test docs)
    file_paths = args.files  # Specific files to process
    persist_dir = args.persist  # Directory to persist vector database
    history_file = args.history  # File to save conversation history
    
    # Only create test directory and sample document if we're using the test directory and no specific files
    if directory == "./test_docs" and not os.path.exists(directory) and not file_paths:
        print(f"Creating test directory: {directory}")
        os.makedirs(directory)
        
        # Create a sample document for testing
        sample_doc = os.path.join(directory, "sample.txt")
        print(f"Creating sample document: {sample_doc}")
        with open(sample_doc, "w") as f:
            f.write("""# Sample Document for RAG Testing
            
This is a sample document for testing Retrieval-Augmented Generation.

## Key Information

- RAG combines retrieval with generation
- It helps ground LLM responses in factual information
- This approach reduces hallucinations
- The implementation uses a vector database for semantic search

## Technical Details

The architecture consists of these components:
1. Document loading and chunking
2. Embedding generation
3. Vector database storage
4. Retrieval mechanism
5. Response generation

You can ask questions about RAG and this system should retrieve this information.
""")
    
    # Initialize the required components
    setup_api_key()
    import_langchain()
    
    # Initialize the LLM
    model = init_llm()
    
    # Choose one of these loading methods:
    try:
        # Determine how to load documents based on args
        if file_paths:
            print(f"Loading specific files: {file_paths}")
            docs = load_documents(file_paths=file_paths)
        # Load from directory
        elif os.path.exists(directory):
            print(f"Loading documents from directory: {directory}")
            # Check if directory has content
            files = os.listdir(directory)
            if not files:
                print(f"WARNING: Directory {directory} exists but is empty!")
                print("Please check the directory path or provide specific files with --files option")
                return
            
            # Print some info about files being loaded
            txt_files = [f for f in files if f.lower().endswith('.txt')]
            if txt_files:
                print(f"Found {len(txt_files)} text files: {', '.join(txt_files[:5])}{' and more...' if len(txt_files) > 5 else ''}")
            else:
                print("No .txt files found directly in the directory. Will search recursively for text files.")
            
            # Load documents
            docs = load_documents(directory_path=directory)
        else:
            print(f"WARNING: Directory {directory} does not exist!")
            print("Please provide a valid directory with --dir or specific files with --files")
            return
        
        # Set up vector store
        vectorstore = setup_vectorstore(docs, persist_directory=persist_dir)
        
        # Create RAG chain
        print("Creating RAG chain...")
        rag_chain = create_rag_chain(vectorstore, model)
        print("RAG chain created successfully")
        
        # Initialize chat
        print("Initializing chat interface...")
        chat = RAGChat(rag_chain, load_history=True, history_file=history_file)
        
        # Display existing conversation with a header
        if len(chat.conversation_history) > 0:
            print("\n" + "="*50)
            print("       PREVIOUS CONVERSATION HISTORY       ")
            print("="*50)
            chat.display_history()
            print("="*50 + "\n")
        
        print("\n\033[1mWelcome to the RAG Chat Terminal!\033[0m")
        print("You can now chat with your documents!")
        print("Type '\033[1mexit\033[0m' to end the conversation.")
        print("Type '\033[1mhistory\033[0m' to view conversation history.")
        print("Type '\033[1mclear\033[0m' to clear conversation history.")
        
        # Interactive chat loop
        while True:
            user_input = input("\n\033[1mYour question:\033[0m ")
            
            # Check for special commands
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'history':
                print("\n" + "="*50)
                print("       CONVERSATION HISTORY       ")
                print("="*50)
                chat.display_history()
                print("="*50)
                continue
            elif user_input.lower() == 'clear':
                chat.conversation_history = []
                save_conversation(chat.conversation_history, history_file)
                print("Conversation history cleared.")
                continue
            elif not user_input.strip():
                continue
                
            # Process the query
            print("Processing your question...\n")
            response = chat.chat(user_input)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nExample usage:")
        print("1. Provide a directory with documents: ragchat --dir /path/to/docs")
        print("2. Provide specific files: ragchat --files doc1.txt doc2.pdf")
        return 1
    
    return 0

def cli_main():
    """Entry point for the CLI command"""
    print("Starting RAGChat CLI...")
    sys.exit(main()) 