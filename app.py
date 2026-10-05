import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
import ollama
from PyPDF2 import PdfReader


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="My Chatbot",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("My Chatbot!!!")


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# --------------------------------------------------
# CHROMADB
# --------------------------------------------------

@st.cache_resource
def get_collection():

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection(
        name="documents"
    )

    return collection


collection = get_collection()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("💬 Chat History")

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    # Display chat history
    if len(st.session_state.messages) == 0:

        st.info("No chat history yet.")

    else:

        for msg in st.session_state.messages:

            if msg["role"] == "user":

                st.markdown("👤 **You**")

                st.write(msg["content"])

            else:

                st.markdown("🤖 **Bot**")

                st.write(msg["content"])

            st.divider()


# --------------------------------------------------
# PDF UPLOAD
# --------------------------------------------------

st.subheader("📄 Upload your PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


# --------------------------------------------------
# PROCESS PDF
# --------------------------------------------------

if uploaded_file is not None:

    # Read PDF
    pdf_reader = PdfReader(uploaded_file)

    text = ""

    for page in pdf_reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    # Check PDF text
    if text.strip() == "":

        st.error("❌ Could not extract text from this PDF.")

    else:

        st.success("✅ PDF loaded successfully!")

        # --------------------------------------------------
        # CREATE CHUNKS
        # --------------------------------------------------

        chunk_size = 500

        chunks = []

        for i in range(0, len(text), chunk_size):

            chunk = text[i:i + chunk_size]

            if chunk.strip():

                chunks.append(chunk)

        st.write(
            f"📚 Total chunks created: {len(chunks)}"
        )

        # --------------------------------------------------
        # CREATE EMBEDDINGS
        # --------------------------------------------------

        embeddings = embedding_model.encode(
            chunks
        )

        # --------------------------------------------------
        # ADD TO CHROMADB
        # --------------------------------------------------

        ids = [
            f"chunk_{i}"
            for i in range(len(chunks))
        ]

        # Delete old data
        try:

            collection.delete(
                ids=collection.get()["ids"]
            )

        except:

            pass

        # Add new data
        collection.add(

            documents=chunks,

            ids=ids,

            embeddings=embeddings.tolist()
        )

        st.success("✅ PDF indexed successfully!")


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask a question:"
)


# --------------------------------------------------
# PROCESS QUESTION
# --------------------------------------------------

if question:

    # Display user question
    with st.chat_message("user"):

        st.write(question)

    # Save user message
    st.session_state.messages.append({

        "role": "user",

        "content": question

    })

    # --------------------------------------------------
    # CREATE QUESTION EMBEDDING
    # --------------------------------------------------

    question_embedding = embedding_model.encode(
        [question]
    )[0].tolist()

    # --------------------------------------------------
    # SEARCH CHROMADB
    # --------------------------------------------------

    results = collection.query(

        query_embeddings=[question_embedding],

        n_results=3
    )

    # --------------------------------------------------
    # GET RELEVANT DOCUMENTS
    # --------------------------------------------------

    documents = results.get(
        "documents",
        [[]]
    )[0]

    if documents:

        context = "\n\n".join(
            documents
        )

    else:

        context = "No relevant information found."

    # --------------------------------------------------
    # CREATE PROMPT
    # --------------------------------------------------

    prompt = f"""
You are a helpful chatbot.

Answer the user's question using the information
provided in the context below.

If the answer is not available in the context,
say that the answer is not available in the uploaded document.

Context:
{context}

User Question:
{question}

Answer:
"""

    # --------------------------------------------------
    # OLLAMA
    # --------------------------------------------------

    try:

        response = ollama.chat(

            model="llama3.2",

            messages=[

                {
                    "role": "user",
                    "content": prompt
                }

            ]

        )

        answer = response["message"]["content"]

    except Exception as e:

        answer = f"""
❌ Ollama Error:

{e}

Make sure Ollama is running and the model is installed.
"""

    # --------------------------------------------------
    # DISPLAY BOT RESPONSE
    # --------------------------------------------------

    with st.chat_message("assistant"):

        st.write(answer)

    # --------------------------------------------------
    # SAVE BOT RESPONSE
    # --------------------------------------------------

    st.session_state.messages.append({

        "role": "assistant",

        "content": answer

    })
