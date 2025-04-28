from setuptools import setup, find_packages

setup(
    name="ragchat",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain-community",
        "langchain-core",
        "python-dotenv",
        "sentence-transformers",
        "chromadb",
        "unstructured",
        "pypdf",
    ],
    entry_points={
        "console_scripts": [
            "ragchat=ragchat.cli:cli_main",
        ],
    },
    description="A Retrieval-Augmented Generation (RAG) chat tool for querying information from documents",
    author="RAG Chat Team",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
) 