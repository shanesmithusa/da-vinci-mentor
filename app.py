import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

st.set_page_config(page_title="Leonardo da Vinci Mentor", layout="centered")
st.title("🎨 Chat with Leonardo da Vinci")

with st.sidebar:
    st.markdown("#### 📄 Upload a Da Vinci Volume (.txt)")
    uploaded_file = st.file_uploader("Upload a mentor file", type="txt")

if uploaded_file:
    with open("temp.txt", "wb") as f:
        f.write(uploaded_file.read())

    loader = TextLoader("temp.txt")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    db = FAISS.from_documents(chunks, OpenAIEmbeddings())
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo"),
        retriever=db.as_retriever()
    )



    query = st.text_input("Ask Leonardo a question:")

    if query:
        response = qa_chain.run(query)
        st.markdown(f"**Da Vinci:** {response}")


