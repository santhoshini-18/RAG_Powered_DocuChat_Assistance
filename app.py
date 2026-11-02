
import streamlit as st
from rag_pipeline import build_vector_store, ask_question

# Page configuration
st.set_page_config(
    page_title="DocuChat AI",
    page_icon="📚",
    layout="wide"
)

# App heading
st.title("📚 DocuChat AI")
st.write("Upload a PDF and ask questions about its content.")

# PDF upload
uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"]
)

# Process uploaded PDF
if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("Process PDF"):
        try:
            with st.spinner("Processing PDF... This may take a few minutes."):
                vector_store = build_vector_store(
                    uploaded_file.getvalue()
                )

                st.session_state.vector_store = vector_store
                st.session_state.chat_history = []
                st.session_state.processed_file = uploaded_file.name

            st.success("PDF processed successfully!")

        except Exception as e:
            st.error(f"PDF processing failed: {e}")

# Display previous chat messages
if "chat_history" in st.session_state:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Question input
question = st.chat_input(
    "Ask a question about your document..."
)

if question:
    st.chat_message("user").write(question)

    if "vector_store" in st.session_state:
        try:
            with st.spinner("Finding your answer..."):
                result = ask_question(
                    st.session_state.vector_store,
                    question
                )

            answer = result["answer"]
            sources = result["sources"]

            # Display answer and unique source pages
            with st.chat_message("assistant"):
                st.write(answer)

                if sources:
                    with st.expander("View source pages"):
                        displayed_pages = set()

                        for source in sources:
                            page_number = source["page"]

                            # Display each page only once
                            if page_number not in displayed_pages:
                                displayed_pages.add(page_number)

                                st.markdown(
                                    f"**Page {page_number}**"
                                )
                                st.write(source["content"])

            # Save chat history
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:
            st.error(f"Could not generate an answer: {e}")

    else:
        st.warning("Please upload and process a PDF first.")
