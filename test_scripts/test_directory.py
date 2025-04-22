import os
import glob
from langchain_community.document_loaders import TextLoader

print("Starting directory test with manual loading...")

# Test manual directory loading
try:
    directory = "./test_docs"
    
    # Manually find all .txt files
    print(f"Finding all txt files in {directory}")
    all_files = glob.glob(os.path.join(directory, "**/*.txt"), recursive=True)
    print(f"Found {len(all_files)} txt files: {all_files}")
    
    # Manually load each file
    all_documents = []
    for file_path in all_files:
        print(f"Loading file: {file_path}")
        loader = TextLoader(file_path)
        docs = loader.load()
        all_documents.extend(docs)
        print(f"  - Loaded {len(docs)} documents")
    
    print(f"\nSuccessfully loaded all documents: {len(all_documents)} total")
    
    # Test using recursive text splitter
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    print("Splitting documents...")
    splits = text_splitter.split_documents(all_documents)
    print(f"Split into {len(splits)} chunks")
    
    # Test with embeddings
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("\nInitializing embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Embeddings initialized")
    
    # Test with vectorstore
    from langchain_community.vectorstores import Chroma
    print("\nCreating vector store...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )
    print("Vector store created successfully")
    
    # Test retrieval
    print("\nTesting retrieval...")
    query = "What is RAG?"
    docs = vectorstore.similarity_search(query)
    print(f"Retrieved {len(docs)} documents for query: '{query}'")
    print(f"First document: {docs[0].page_content[:100]}...")
    
    print("\nAll tests completed successfully!")
    
except Exception as e:
    import traceback
    print(f"Error: {str(e)}")
    traceback.print_exc() 