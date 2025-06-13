# 🦙 LlamaIndex Tutorials Collection

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)]()
[![Maintainer](https://img.shields.io/static/v1?label=Yevhen%20Ruban&message=Maintainer&color=red)](mailto:yevhen.ruban@extrawest.com)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub release](https://img.shields.io/badge/release-v1.0.0-blue)

This repository contains a collection of demonstration projects showcasing various capabilities and applications of LlamaIndex, a powerful data framework for building LLM applications with custom data. Each tutorial focuses on a specific aspect of LlamaIndex functionality, ranging from basic usage to advanced features like RAG (Retrieval Augmented Generation), chatbots, and function-calling agents.

## 📚 Tutorials Overview

### 🚀 Simple Demo
A minimal introduction to LlamaIndex's core functionality.

**🧠 What You'll Learn:**
- Basic setup of LlamaIndex
- Loading documents from a directory
- Creating a vector store index
- Running simple queries against your data

**⚙️ How to Run:**
```bash
# Install dependencies
pip install -r simple_demo/requirements.txt

# Create a data directory and add some text files
mkdir -p data
# Add some text files to the data directory
python simple_demo/llama_index_simple_demo.py
```

**📦 Dependencies:**
Requirements are specified in `simple_demo/requirements.txt`:
```
llama-index>=0.9.48
llama-index-core>=0.9.48
llama-index-readers-file>=0.1.0
python-dotenv>=1.0.0
pathlib>=1.0.1
```

### 🔍 RAG System: Retrieval Augmented Generation with LlamaIndex and Open-Source Models

This tutorial demonstrates how to create a RAG system using locally running open-source models.

**🧠 What You'll Learn:**
- Setting up LlamaIndex with local models (Mistral)
- Creating and persisting vector indices
- Configuring custom embedding models
- Implementing RAG with different storage backends (file-based and PostgreSQL)
- Optimizing chunk size and overlap for better retrieval

**📁 File Structure:**
- `llama_rag_system.py`: RAG implementation with file-based storage
- `llama_rag_system_psql.py`: RAG implementation with PostgreSQL vector store
- `requirements.txt`: Dependencies list

**⚙️ How to Run:**
```bash
# Install dependencies
pip install -r rag_system/requirements.txt

# First download the Mistral model
mkdir -p ../llm
# Download Mistral model to ../llm/mistral-7b-instruct-v0.1.Q2_K.gguf

# For file-based storage
mkdir -p ../data ../index
# Add PDF files to ../data directory
python rag_system/llama_rag_system.py

# For PostgreSQL-based storage (requires PostgreSQL running)
# Ensure PostgreSQL is running with the right credentials
python rag_system/llama_rag_system_psql.py
```

**📦 Dependencies:**
Requirements are specified in `rag_system/requirements.txt`:
```
# Core dependencies
transformers>=4.37.0
torch>=2.1.0
llama-index>=0.9.48
llama-index-core>=0.9.48
llama-cpp-python>=0.2.23
sentence-transformers>=2.2.2
llama-index-embeddings-huggingface>=0.1.0

# PostgreSQL support (for llama_rag_system_psql.py)
psycopg2-binary>=3.2.0
sqlalchemy>=2.0.27
llama-index-vector-stores-postgres>=0.1.0

# Optional utilities
tiktoken>=0.5.2
python-dotenv>=1.0.0
pathlib>=1.0.1
tqdm>=4.65.0
```

### 🤖 Chatbot: Building an Interactive Chatbot with LlamaIndex

This tutorial demonstrates how to build an interactive chatbot that can analyze multiple documents and answer questions using OpenAI models.

**🧠 What You'll Learn:**
- Setting up a document processing pipeline using LlamaIndex
- Creating and persisting separate indices for different document sources
- Implementing a multi-document query engine
- Building a ReActAgent chatbot with OpenAI
- Handling asynchronous operations

**📁 File Structure:**
- `llama_chatbot.py`: Complete chatbot implementation
- `requirements.txt`: Dependencies list

**⚙️ How to Run:**
```bash
# Install dependencies
pip install -r chatbot/requirements.txt

# Create necessary directories
mkdir -p ../data ../storage

# Add HTML files named UBER_YYYY.html to the data directory
# (Where YYYY represents years like 2019, 2020, 2021, 2022)

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Run the chatbot
python chatbot/llama_chatbot.py
```

**📦 Dependencies:**
```
llama-index>=0.10.0
llama-index-core>=0.10.0
llama-index-readers-file>=0.1.0
llama-index-embeddings-openai>=0.1.0
llama-index-agent-openai>=0.1.0
llama-index-llms-openai>=0.1.0
llama-index-question-gen-openai>=0.1.0
unstructured>=0.10.0
python-dotenv>=1.0.0
tqdm>=4.66.2
numpy>=1.26.4
```

### ⚡ Function Calling: Building a Function Calling Mistral Agent with LlamaIndex

This tutorial demonstrates how to build an AI agent with function-calling capabilities using Mistral AI and LlamaIndex.

**🧠 What You'll Learn:**
- Setting up a Function Agent with Mistral AI
- Implementing custom tools as Python functions
- Managing context and conversation memory
- Integrating RAG with agent architecture
- Implementing retry logic for API rate limits
- Logging agent interactions

**📁 File Structure:**
- `llama_function_calling.py`: Complete function-calling agent implementation
- `requirements.txt`: Dependencies list

**⚙️ How to Run:**
```bash
# Install dependencies
pip install -r function_calling/requirements.txt

# Set your Mistral API key
export MISTRAL_API_KEY="your-mistral-api-key-here"

# Create data directory structure
mkdir -p data/10k

# Add Uber 10-K PDF to data/10k/uber_2021.pdf

# Run the function calling agent demo
python function_calling/llama_function_calling.py
```

**📦 Dependencies:**
```
llama-index>=0.10.0
llama-index-core>=0.10.0
llama-index-llms-llama-cpp>=0.4.0
llama-index-llms-mistralai  # Not listed but required
llama-index-embeddings-mistralai  # Not listed but required
transformers
torch
numpy>=1.26.4
tqdm>=4.66.2
python-dotenv
```

## 🛠️ General Setup Instructions

1. **🐍 Python Environment**:
   ```bash
   # Create and activate a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **📥 Install the base dependencies**:
   ```bash
   pip install llama-index python-dotenv
   ```

3. **🧩 Model Setup**:
   - For tasks using local models (like rag_system), download the required model files to the specified directory.
   - For tasks using API-based models (like chatbot and function_calling), obtain the necessary API keys.

4. **📊 Data Preparation**:
   - Each tutorial expects specific data structures. Make sure to prepare the necessary data files as mentioned in each tutorial's section.

5. **📂 Directory Structure**:
   - The repository assumes a specific directory structure. Some paths are relative to the parent directory, so make sure to maintain the expected structure.

## 📝 Notes

- The tutorials incrementally increase in complexity from simple_demo to function_calling.
- You can modify the code to use different models or data sources based on your needs.
- For the best experience, follow the tutorials in order to build a comprehensive understanding of LlamaIndex capabilities.

Developed by [extrawest](https://extrawest.com/). Software development company
