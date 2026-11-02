# RAG_Powered_DocuChat_Assistance
DocuChat is a Retrieval-Augmented Generation (RAG) application that lets users ask questions about PDF documents and receive answers grounded in the document content.
Features
- Load and process PDF documents.
- Split extracted text into smaller chunks for retrieval.
- Generate semantic embeddings using Hugging Face.
- Store and retrieve document embeddings with ChromaDB.
- Use Google Gemini to generate context-aware answers from retrieved content.
- Provide an interactive interface using Streamlit.
Tech Stack
- Language: Python
- RAG framework: LangChain
- PDF loading: PyPDFLoader
- Text splitting: RecursiveCharacterTextSplitter
- Embeddings: Hugging Face – sentence-transformers/all-mpnet-base-v2
- Vector database: ChromaDB
- LLM: Google Gemini
- UI: Streamlit
- Environment variables: python-dotenv
How It Works
1. Load PDF: PyPDFLoader extracts text from the PDF.
2. Split text: RecursiveCharacterTextSplitter divides the text into manageable chunks.
3. Create embeddings: The Hugging Face embedding model converts each chunk into a vector representation.
4. Store vectors: ChromaDB stores the embeddings for similarity-based retrieval.
5. Retrieve context: When a user asks a question, the retriever finds relevant document chunks.
6. Generate an answer: Google Gemini uses the retrieved context to generate a response.
7. Display response: Streamlit presents the interaction through the app interface.
Project Structure
The exact filenames may differ depending on your implementation. A typical structure is:
DocuChat/
├── app.py                 # Streamlit application
├── rag_pipeline.py        # PDF processing, retrieval, and answer generation
├── requirements.txt       # Python dependencies
├── .env                   # API key (do not commit this file)
├── .gitignore
└── README.md
Getting Started
1. Clone the repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd DocuChat
2. Create and activate a virtual environment
Windows:
python -m venv venv
venv\Scripts\activate
macOS / Linux:
python3 -m venv venv
source venv/bin/activate
3. Install dependencies
If your project includes a requirements.txt file:
pip install -r requirements.txt
Otherwise, install the packages used by your implementation and save them to a requirements file.
4. Configure the Gemini API key
Create a .env file in the project root:
GEMINI_API_KEY=your_gemini_api_key
Replace the placeholder with your own API key. Keep the .env file private and never upload it to GitHub.
5. Run the application
If app.py is your Streamlit entry point:
streamlit run app.py
Open the local URL shown in the terminal.
Example Use
1. Start the DocuChat application.
2. Upload a PDF document through the interface.
3. Enter a question about the document.
4. DocuChat retrieves relevant text and uses Gemini to generate an answer based on that context.
Security
- Store API keys in environment variables or a local .env file.
- Add .env, virtual environments, and generated database files to .gitignore where appropriate.
- Do not commit API keys, credentials, or private documents to a public repository.
Future Improvements
- Display source page numbers or retrieved passages alongside answers.
- Support multiple PDF documents in one session.
- Add conversation history.
- Improve document-processing feedback and error handling.
Author
Santhoshini Amberkar
- GitHub: santhoshini-18
- LinkedIn: Santhoshini Amberkar
DocuChat was built to explore practical applications of Retrieval-Augmented Generation, semantic search, vector databases, and Large Language Models.
