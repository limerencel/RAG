import os
import time
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

print("Starting minimal test...")

# Test document loading
print("Testing document loading...")
directory = "./test_docs"
if not os.path.exists(directory):
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
""")

try:
    print(f"Loading documents from directory: {directory}")
    txt_files = os.listdir(directory)
    print(f"Files in directory: {txt_files}")
    
    loader = DirectoryLoader(directory, glob="**/*.txt")
    print("Created directory loader")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    
    # Split documents into chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} chunks")
    
    # Test embedding creation
    print("\nTesting embedding creation...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Created HuggingFace embeddings model")
    
    # Test vectorstore creation
    print("\nTesting vectorstore creation...")
    start_time = time.time()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    elapsed_time = time.time() - start_time
    print(f"Created vectorstore in {elapsed_time:.2f} seconds")
    
    print("\nAll tests completed successfully!")
    
except Exception as e:
    import traceback
    print(f"Error: {str(e)}")
    traceback.print_exc() 