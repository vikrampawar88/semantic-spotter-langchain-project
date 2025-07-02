# Semantic Spotter: A RAG-based Insurance Search System

## 1. Overview

This project showcases how to build a Retrieval-Augmented Generation (RAG) system tailored to the insurance industry using [LangChain](https://python.langchain.com/docs/introduction/).

## 2. Objective

The primary aim is to develop a powerful generative search engine that can accurately respond to queries by referencing information from various insurance policy documents.

## 3. Data Source

The policy documents used for this project are located in the [Policy_Documents](./Policy_Documents) directory.

## 4. Methodology

LangChain is an open-source framework that simplifies the process of creating applications powered by large language models (LLMs). It provides a variety of tools, integrations, and abstractions to streamline development.

LangChain supports both Python and JavaScript/TypeScript, emphasizing a modular and composable architecture. With built-in support for different LLMs (like OpenAI, Cohere, Hugging Face), it enables developers to rapidly build and customize intelligent applications that can pull context from external sources.

### Key Components of LangChain:

- **LLM Interfaces**: Abstractions to interact with different LLM providers
- **Chains**: Sequences of calls and logic used to handle specific tasks
- **Retrievers**: Retrieve documents based on user queries
- **Agents**: Choose tools dynamically based on task requirements
- **Memory**: Maintain state across interactions
- **Callbacks**: Capture logs or intermediate outputs during execution

LangChain includes both low-level components (like document loaders and vector stores) and high-level pre-built chains for specific use cases, allowing flexibility and ease of use.

## 5. Implementation Layers

- **PDF Reading & Parsing**: Policy documents are ingested using LangChain's [PyPDFDirectoryLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PyPDFDirectoryLoader.html), which processes all PDF files in a specified folder.

- **Chunking Documents**: To enhance retrieval effectiveness, documents are split using [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/how_to/recursive_text_splitter/), which maintains semantic cohesion by splitting based on a prioritized list of separators (e.g., paragraph breaks, sentences, words).

- **Text Embeddings**: Embeddings are created using LangChain’s [OpenAIEmbeddings](https://python.langchain.com/docs/integrations/text_embedding/openai/) integration. These embeddings convert text into vector format, allowing similarity comparisons and semantic search.

- **Embedding Storage**: The generated embeddings are stored in [ChromaDB](https://docs.trychroma.com/), utilizing LangChain's [CacheBackedEmbeddings](https://python.langchain.com/api_reference/langchain/embeddings/langchain.embeddings.cache.CacheBackedEmbeddings.html) for efficient reuse and performance.

- **Retriever Mechanism**: Retrievers serve as interfaces that fetch relevant documents in response to natural language queries. This project uses the [VectorStoreRetriever](https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.base.VectorStoreRetriever.html) to connect with the vector store.

- **Re-Ranking with Cross Encoders**: To improve the quality of the search results, the retrieved documents are re-ranked using a cross-encoder model ([BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)) via LangChain’s [HuggingFaceCrossEncoder](https://python.langchain.com/api_reference/community/cross_encoders/langchain_community.cross_encoders.huggingface.HuggingFaceCrossEncoder.html).

- **RAG Chains**: LangChain chains are used to link all components—embedding, retrieval, re-ranking, and LLM generation. A pre-built prompt template from LangChain Hub, `rlm/rag-promp`, is used within the RAG pipeline.

## 6. Architecture Overview

![Architecture 1](./images/arch1.png)  
![Architecture 2](./images/arch2.png)

## 7. Requirements

- Python ≥ 3.7
- langchain == 0.3.13
- An OpenAI API key placed in a text file named `API_Key.txt` for authentication with the LLM service

## 8. How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/vikrampawar88/semantic-spotter-langchain-project.git
2. Open the Jupyter notebook:
semantic-spotter-langchain-project.ipynb

3. Run all the cells to execute the pipeline.