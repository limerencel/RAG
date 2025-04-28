#!/bin/bash
# Install the RAGChat CLI tool

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not found. Please install Python 3 and try again."
    exit 1
fi

# Install the package
echo "Installing RAGChat CLI tool..."
python3 -m pip install -e .

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "Installation successful!"
    echo "You can now use the 'ragchat' command from your terminal."
    echo
    echo "Example usage:"
    echo "  ragchat --dir /path/to/documents"
    echo "  ragchat --files doc1.txt doc2.pdf"
    echo
    echo "Don't forget to create a .env file with your xAI API key:"
    echo "XAI_API_KEY=your-api-key-here"
else
    echo "Installation failed. Please check the error messages above."
    echo "Try using: python3 -m pip install --upgrade pip"
    echo "Then run the install script again."
fi 