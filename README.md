# RAG-Based-Healthcare-Chatbot
```bash
This project is an AI-powered healthcare chatbot that answers medical-related questions using a Retrieval-Augmented Generation (RAG) architecture.

Instead of relying only on a language model, the chatbot retrieves relevant information from healthcare documents and then generates a response based on that knowledge. This helps improve the accuracy and reliability of answers.

The application is built with Python, LangChain, Ollama, Pinecone, and Streamlit, and it demonstrates how modern AI systems combine large language models with vector databases.

This project is designed as a portfolio-level AI application to showcase skills in LLM integration, vector databases, semantic search, and AI application development.
```

```bash
Features
AI chatbot that answers healthcare-related questions
Retrieval-Augmented Generation (RAG) architecture
Semantic search using a vector database
Document-based knowledge retrieval
Local LLM support using Ollama
Interactive web interface built with Streamlit
Chat history support
Source-based responses from documents
```

```bash
Technologies Used
- Python – Main programming language
- LangChain – Framework for building LLM applications
- Ollama – Runs large language models locally
- Pinecone – Vector database for storing embeddings
- Sentence Transformers – Generates embeddings for documents
- Streamlit – Creates the web interface
```


# How to run ?
### steps :
clone the repository

```bash
git clone : https://github.com/Chidara-Supriya/RAG-Based-Healthcare-Chatbot.git
```
### create a conda environment after opening the repository

```bash
conda create -n healthcarebot python=3.10 -y
```

```bash
conda activate healthcarebot
```

###  install the requirements

```bash
pip install -r requirement.txt
```

### Install Ollama
``bash
ollama push llama3
```

### Environment Variables
Create a .env file in the root directory and add your Pinecone API key.
```bash
PINECONE_API_KEY=your_api_key_here
PINECONE_INDEX_NAME=healthcare-chatbot
```

### Run the Chatbot Application

```bash
streamlit run app.py
```

### Then open your browser and go to:
```bash
http://localhost:
```
### Deployment
- This project can de deployed on several platform such as:
1.Streamlit CLoud
2.Hugging Face Spaces
3.Render
4.AWS/GCP/Azure

For portfolio projects ,Streamlit Cloud is the easiest option

### Future Improvements
Possible enhancements for this project include:

Adding more healthcare datasets

Voice-based chatbot interaction

Multi-language support

Doctor appointment integration

Advanced medical knowledge graphs



