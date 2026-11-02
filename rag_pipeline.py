
import os
import tempfile

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing. Check your .env file."
    )

# --------------------------------------------------
# 2. Initialize Gemini
# --------------------------------------------------

llm = init_chat_model(
    "google_genai:gemini-3.5-flash-lite",
    api_key=API_KEY
)

# --------------------------------------------------
# 3. Initialize the embedding model
# --------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------------------------
# 4. Build vector store from uploaded PDF
# --------------------------------------------------

def build_vector_store(pdf_bytes):
    """Load a PDF, split it into chunks, and create a vector store."""

    pdf_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False
        ) as temp_file:
            temp_file.write(pdf_bytes)
            pdf_path = temp_file.name

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        if not documents:
            raise ValueError(
                "The uploaded PDF contains no readable pages."
            )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = text_splitter.split_documents(documents)

        if not chunks:
            raise ValueError(
                "No readable text was found in this PDF."
            )

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model
        )

        return vector_store

    finally:
        if pdf_path and os.path.exists(pdf_path):
            os.remove(pdf_path)


# --------------------------------------------------
# 5. Ask questions about the PDF
# --------------------------------------------------

def ask_question(vector_store, user_query, k=4):
    """Retrieve relevant PDF chunks and generate a grounded answer."""

    if not user_query or not user_query.strip():
        return {
            "answer": "Please enter a question.",
            "sources": []
        }

    retrieved_docs = vector_store.similarity_search(
        user_query.strip(),
        k=k
    )

    # Debug information: visible in the VS Code terminal.
    for doc in retrieved_docs:
        print(
            "PDF zero-based page index:",
            doc.metadata.get("page"),
            "| Chunk preview:",
            doc.page_content[:100]
        )

    if not retrieved_docs:
        return {
            "answer": (
                "I couldn't find relevant information "
                "in the uploaded document."
            ),
            "sources": []
        }

    context_parts = []

    for doc in retrieved_docs:
        # PyPDFLoader page indexes start at zero.
        pdf_page_index = doc.metadata.get("page", 0)

        context_parts.append(
            f"[PDF viewer page {pdf_page_index + 1}]\n"
            f"{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    system_message = SystemMessage(
        content=(
            "You are DocuChat, a PDF question-answering assistant.\n"
            "Answer the user's question using only the supplied "
            "document context.\n"
            "Give a clear, accurate explanation with relevant "
            "details from the retrieved passages.\n"
            "Do not invent facts or claim unsupported information.\n"
            "If the retrieved context is insufficient, say so.\n"
            "Page references in the context are PDF viewer page "
            "numbers, not necessarily the printed page numbers "
            "inside the document.\n\n"
            f"DOCUMENT CONTEXT:\n{context}"
        )
    )

    human_message = HumanMessage(
        content=user_query.strip()
    )

    response = llm.invoke(
        [system_message, human_message]
    )

    answer = getattr(response, "text", None)

    if not answer:
        content = response.content

        if isinstance(content, str):
            answer = content

        elif isinstance(content, list):
            text_parts = []

            for block in content:
                if isinstance(block, dict):
                    text_value = block.get("text", "")
                    if text_value:
                        text_parts.append(text_value)
                elif isinstance(block, str):
                    text_parts.append(block)

            answer = "\n".join(text_parts)

        else:
            answer = str(content)

    sources = []

    for doc in retrieved_docs:
        pdf_page_index = doc.metadata.get("page", 0)

        sources.append({
            # Actual PDF viewer page number.
            "page": pdf_page_index + 1,

            # Original zero-based index for debugging.
            "page_index": pdf_page_index,

            "content": doc.page_content
        })

    return {
        "answer": answer,
        "sources": sources
    }
