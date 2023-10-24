import streamlit as st
import pickle
from PyPDF2 import PdfReader
from langchain.llms.openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def app():
    st.header("Chat with your book")
    pdf_file = st.file_uploader("Upload your book", type="pdf")

    IS_READ = "is_read"
    st.session_state[IS_READ] = False
    
    vector_store: FAISS = None

    if pdf_file and not st.session_state[IS_READ]:
        pdf_name = pdf_file.name[:-4]
        pdf_read = PdfReader(pdf_file)

        text_book = ""

        with st.spinner("Read book..."):
            for text in pdf_read.pages:
                text_book += text.extract_text()

        st.info("Read book complete...")
        st.session_state[IS_READ] = True

        with st.spinner("Create chunk"):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )

            text_chunks = text_splitter.split_text(text_book)

        st.info("Create chunk complete...")

        if not os.path.exists("embeddings"):
            os.mkdir("embeddings")

        with st.spinner("Create embedding"):
            if os.path.exists(f"embeddings/{pdf_name}.pkl"):
                with open(f"embeddings/{pdf_name}.pkl", "rb") as f:
                    vector_store = pickle.load(f)
            else:
                openai_embedding = OpenAIEmbeddings()
                vector_store = FAISS.from_texts(text_chunks, openai_embedding)

                with open(f"embeddings/{pdf_name}.pkl", "wb") as f:
                    pickle.dump(vector_store, f)

        st.info("Embedding complete...")

    if vector_store and st.session_state[IS_READ]:
        with st.container():
            question = st.text_input("Ask anything to your book:")

            if question:
                docs = vector_store.similarity_search(query=question, k=5)
                llm = OpenAI()
                chain = load_qa_chain(llm=llm, chain_type="stuff")

                response = chain.run(input_documents=docs, question=question)

                st.markdown("**Your book answer:**")
                st.markdown(f""":blue[{response}]""",)

if __name__ == "__main__":
    app()
