import os
from langchain_community.document_loaders import TextLoader

print("Starting very simple test...")

# Test direct text loading
try:
    # Create a sample document for testing if it doesn't exist
    directory = "./test_docs"
    sample_doc = os.path.join(directory, "sample.txt")
    
    if not os.path.exists(directory):
        print(f"Creating test directory: {directory}")
        os.makedirs(directory)
        
    if not os.path.exists(sample_doc):
        print(f"Creating sample document: {sample_doc}")
        with open(sample_doc, "w") as f:
            f.write("This is a test document.")
    
    print(f"Loading file directly: {sample_doc}")
    loader = TextLoader(sample_doc)
    print("Created text loader")
    
    documents = loader.load()
    print(f"Successfully loaded document: {len(documents)} documents")
    print(f"Document content preview: {documents[0].page_content[:50]}...")
    
    print("\nTest completed successfully!")
    
except Exception as e:
    import traceback
    print(f"Error: {str(e)}")
    traceback.print_exc() 