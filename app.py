import os
from dotenv import load_dotenv
load_dotenv()

import openai
import streamlit as st
from PyPDF2 import PdfReader
from io import StringIO

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.7,model_name='gpt-3.5-turbo')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def main():
    st.title("PDF to Text")
    st.header("Upload a PDF file and ask any question about it.")

    pdf = st.file_uploader("Upload a PDF file", type=["pdf"], accept_multiple_files=False)
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            text_splitter = CharacterTextSplitter(
                separator='\n',
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
                )
            chunks = text_splitter.split_text(text)
        st.write("PDF file uploaded successfully")

        # we will create embeddings for each chunk

        embadding = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embadding)
        st.write("Embeddings created successfully")
        query = st.text_input("Ask a question about the PDF file")
        if query:
            st.write("Searching for the answer...")
            docs = knowledge_base.similarity_search(query)
            # st.write(result)
            chain = load_qa_chain(llm,chain_type='stuff')
            response = chain.run(input_documents=docs, question=query)
            st.write(response)


    else:
        st.write("Please upload a PDF file")




if __name__ == "__main__":
    main()