from sentence_transformers import SentenceTransformer
import chromadb
import ollama
import streamlit as st
st.set_page_config(
    page_title="My Chatbot",
    page_icon="🤖",
    layout="wide"
)


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()
client = chromadb.PersistentClient(
    path="./chroma_db"
)
collection = client.get_or_create_collection(
    name="My_Documents"
)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_text" not in st.session_state:
    st.session_state.document_text = ""
if "document_name" not in st.session_state:
    st.session_state.document_name = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "key_features" not in st.session_state:
    st.session_state.key_features = ""
st.title(":rainbow[*My Chatbot!!!*]")
with st.sidebar:
    st.header("⚙️ Chat Settings")
    personalities = {
        "Professor👨‍🏫":
        "You are an IIT professor. Explain topics using correct terminology. Give clear and simple answers.give 2-3 lines only.",
        "Friendly👥":
        "You are a friendly AI assistant. Explain things in simple and easy language."
    }
    personality = st.selectbox(
        "Select a personality",
        list(personalities.keys())
    )
    top_k = st.slider(
        "Select no of top results",
        min_value=1,
        max_value=5,
        value=3
    )
    st.write("📁 Upload a file")

    uploaded_file = st.file_uploader(
        "Upload a TXT file",
        type=["txt"]
    )
    if uploaded_file is not None:
        if st.session_state.document_name != uploaded_file.name:
            try:
                text = uploaded_file.read().decode(
                    "utf-8"
                )
                st.session_state.document_text = text
                st.session_state.document_name = (
                    uploaded_file.name
                )
                st.session_state.summary = ""
                st.session_state.key_features = ""
                chunk_size = 500
                chunk_overlap = 100
                step = (
                    chunk_size -
                    chunk_overlap
                )
                for i in range(
                    0,
                    len(text),
                    step
                ):
                    chunk = text[
                        i:i + chunk_size
                    ]
                    if chunk.strip():
                        chunks.append(chunk)
                embeddings = model.encode(
                    chunks
                )
                ids = []
                for i in range(
                    len(chunks)
                ):
                    ids.append(
                        f"{uploaded_file.name}_{i}"
                    )
                metadatas = []
                for i in range(
                    len(chunks)
                ):
                    metadatas.append(
                        {
                            "source":
                            uploaded_file.name,

                            "chunk":
                            i
                        }
                    )
                collection.upsert(
                    ids=ids,
                    documents=chunks,
                    embeddings=embeddings.tolist(),
                    metadatas=metadatas
                )
                st.success(
                    f"Uploaded: {uploaded_file.name}"
                )
                st.info(
                    f"{len(chunks)} document chunks created."
                )
            except Exception as e:
                st.error(
                    f"Error processing file: {e}"
                )
        with st.expander("👁️ Preview"):
            st.text(
                st.session_state.document_text[
                    :5000
                ]
            )
    st.divider()
    st.subheader("💬 Chat History")
    if st.session_state.messages:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f"👤 **You:** {msg['content']}"
                )
            elif msg["role"] == "assistant":
                st.markdown(
                    f"🤖 **AI:** {msg['content']}"
                )
    else:
        st.info(
            "No chat history yet."
        )
    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()
tab1, tab2, tab3 = st.tabs(
    [
        "💬 Chat",
        "📝 Summarize",
        "⭐ Key Features"
    ]
)
with tab1:
    st.header("💬 Chat")
    for msg in st.session_state.messages:
        with st.chat_message(
            msg["role"]
        ):
            st.write(
                msg["content"]
            )
    question = st.chat_input(
        "Ask a question..."
    )
    if question:
        with st.chat_message("user"):

            st.write(question)
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )
        if st.session_state.document_text:
            with st.chat_message(
                "assistant"
            ):
                with st.spinner(
                    "Searching document..."
                ):
                    try:
                        q_embedding = model.encode(
                            question
                        )
                        results = collection.query(
                            query_embeddings=[
                                q_embedding.tolist()
                            ],
                            n_results=top_k,
                            where={
                                "source":
                                st.session_state.document_name
                            }
                        )
                        if (
                            results
                            and
                            results.get("documents")
                            and
                            results["documents"][0]
                        ):
                            retrieved_chunks = (
                                results["documents"][0]
                            )
                            context = "\n\n".join(
                                retrieved_chunks
                            )
                            prompt = f"""
You are an AI assistant.

Answer the question using ONLY
the information from the uploaded document.

If the answer is not available
in the document, say:

"I could not find the answer
in the uploaded document."

Give the answer in simple
and clear language.

Question:

{question}

Document Context:

{context}

Answer:
"""
                            response = ollama.chat(
                                model="llama3.2:3b",
                                messages=[
                                    {
                                        "role":
                                        "system",

                                        "content":
                                        personalities[
                                            personality
                                        ]
                                    },
                                    {
                                        "role":
                                        "user",

                                        "content":
                                        prompt
                                    }
                                ]
                            )
                            answer = (
                                response[
                                    "message"
                                ][
                                    "content"
                                ]
                            )

                        else:

                            answer = (
                                "I could not find "
                                "relevant information "
                                "in the uploaded document."
                            )

                    except Exception as e:

                        answer = (
                            f"Error: {e}"
                        )
                st.write(answer)
            st.session_state.messages.append(
                {
                    "role":
                    "assistant",

                    "content":
                    answer
                }
            )
        else:
            with st.chat_message(
                "assistant"
            ):
                with st.spinner(
                    "Thinking..."
                ):

                    try:

                        response = ollama.chat(

                            model="llama3.2:3b",

                            messages=[

                                {
                                    "role":
                                    "system",

                                    "content":
                                    personalities[
                                        personality
                                    ]
                                }

                            ]
                            +
                            st.session_state.messages

                        )

                        answer = (
                            response[
                                "message"
                            ][
                                "content"
                            ]
                        )

                    except Exception as e:

                        answer = (
                            f"Error: {e}"
                        )

                st.write(answer)
            st.session_state.messages.append(
                {
                    "role":
                    "assistant",

                    "content":
                    answer
                }
            )
with tab2:

    st.header("📝 Summarize")

    if st.session_state.document_text:

        st.write(
            "Generate a simple summary "
            "of the uploaded document."
        )

        if st.button(
            "📝 Generate Summary",
            use_container_width=True
        ):

            with st.spinner(
                "Generating summary..."
            ):

                try:

                    document_text = (
                        st.session_state.document_text[
                            :15000
                        ]
                    )

                    summary_prompt = f"""
Summarize the following document.

Give the summary in simple language.

Use clear bullet points.

Mention the important ideas,
concepts and information.

Document:

{document_text}

Summary:
"""

                    response = ollama.chat(

                        model="llama3.2:3b",

                        messages=[

                            {
                                "role":
                                "user",

                                "content":
                                summary_prompt
                            }

                        ]
                    )

                    st.session_state.summary = (
                        response[
                            "message"
                        ][
                            "content"
                        ]
                    )

                except Exception as e:

                    st.error(
                        f"Error: {e}"
                    )
        if st.session_state.summary:

            st.subheader(
                "📄 Document Summary"
            )

            st.write(
                st.session_state.summary
            )

    else:

        st.info(
            "Please upload a document first."
        )

with tab3:

    st.header("⭐ Key Features")

    if st.session_state.document_text:

        st.write(
            "Generate the important "
            "concepts and key points "
            "from the uploaded document."
        )

        if st.button(
            "⭐ Generate Key Features",
            use_container_width=True
        ):

            with st.spinner(
                "Finding key features..."
            ):

                try:

                    document_text = (
                        st.session_state.document_text[
                            :15000
                        ]
                    )

                    feature_prompt = f"""
Read the following document.

Identify the important concepts,
features and key points.

Give the answer as clear
bullet points.

Document:

{document_text}

Key Features:
"""

                    response = ollama.chat(

                        model="llama3.2:3b",

                        messages=[

                            {
                                "role":
                                "user",

                                "content":
                                feature_prompt
                            }

                        ]
                    )

                    st.session_state.key_features = (
                        response[
                            "message"
                        ][
                            "content"
                        ]
                    )

                except Exception as e:

                    st.error(
                        f"Error: {e}"
                    )

        if st.session_state.key_features:

            st.subheader(
                "🔑 Important Features"
            )

            st.write(
                st.session_state.key_features
            )
    else:

        st.info(
            "Please upload a document first."
        )
