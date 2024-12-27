# LangChain ChatBot - Chat with Multiple Documents 🤖📚
## 🌟Overview
This project is a **Streamlit-based Conversational AI application** designed to handle queries across multiple types of documents efficiently. Leveraging the **Retrieval-Augmented Generation (RAG)** approach, the chatbot integrates **FAISS** for vector similarity search and **HuggingFace LLMs** for natural language processing. This approach eliminates the need for costly fine-tuning, making it scalable and efficient for real-world applications. The RAG chatbot efficiently combines document retrieval with language generation, 
providing accurate and contextually relevant answers.


## 🚀Key Features 
- **Multi-Document Support**: Processes PDFs, Word documents, PowerPoint presentations, and plain text files.
- **Streamlit Interface**: User-friendly interface for seamless interaction.
- **RAG Architecture**: Combines retrieval-based techniques with generative AI for accurate and context-aware responses.
- **Efficient Vector Search**: Utilizes FAISS to handle large-scale document indexing and similarity matching.
- **Pre-trained HuggingFace LLMs**: Generates conversational and contextually relevant responses.

## 💡Advantages of RAG over Fine-Tuning
- **Cost Efficiency**: Avoids the need for expensive training of LLMs.
- **Flexibility**: Adapts to dynamic data changes without retraining.
- **Scalability**: Handles diverse document types and large datasets with ease.

---


## 🛠️Setup Instructions 

### 1. Clone the Repository
```bash
git clone https://github.com/NidaIslam/LangChain_ChatBot.git
cd LangChain_ChatBot
```

### 2. Open the Colab Notebook 📒
1. Navigate to the provided Colab notebook in the repository.
2. Open the notebook in Google Colab.

### 3. Install Required Packages
Run the notebook cell to install all necessary Python dependencies:
```python
!pip install -r requirements.txt
```

### 4. Configure HuggingFace API Token 🔑
Add your HuggingFace API token in the designated notebook cell:
```python
import os
import getpass

# Enter your HuggingFace API that has read permissions
huggingface_api_key = getpass.getpass("Enter your HuggingFace API key:")

if huggingface_api_key:
    # Set the HuggingFace API token as an environment variable
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key
    print("API token securely set!")
```

### 5. Generate `app.py`
Run the cell in the notebook to write the `app.py` file. This file contains the code for the Streamlit application.

## 📝How to run Streamlit app in Google Colab Notebook? 
### 6. Install LocalTunnel and Serve the Streamlit App 🌐
Install LocalTunnel in the notebook:
```python
!npm install localtunnel
```
Then run the Streamlit application using LocalTunnel to expose it as a public URL:
```python
!streamlit run app.py &>/content/logs.txt & npx localtunnel --port 8501 & curl https://loca.lt/mytunnelpassword
```
It will provide you the URL and Network address that you need to enter to accesss the app in browser when open the URL link.  


## 🖥️Usage
1. Upload your documents (PDFs, DOCX, PPTX, or TXT) via the Streamlit app interface.
2. Interact with the app by asking questions related to the uploaded documents.
3. Receive accurate, context-aware responses in real time.

## How It Works
1. **Document Parsing**: Extracts text from uploaded files using PyPDF2, python-docx, and python-pptx.
2. **Text Chunking**: Splits long texts into manageable chunks for efficient processing.
3. **Vectorization**: Embeds text chunks using Sentence Transformers from HuggingFace.
4. **Vector Store**: Stores embeddings in a FAISS index for fast similarity search.
5. **Conversational Chain**: Uses the RAG framework to retrieve relevant document chunks and generate responses using HuggingFace LLMs.

## License
This project is licensed under the [MIT License](LICENSE).

## 🙌Acknowledgments
Special thanks to:
- [HuggingFace](https://huggingface.co/) for providing pre-trained LLMs.
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search.
- [Streamlit](https://streamlit.io/) for an intuitive UI framework.

---
Feel free to contribute to the project or report issues by opening an [issue](https://github.com/NidaIslam/LangChain_ChatBot/issues) on GitHub.
