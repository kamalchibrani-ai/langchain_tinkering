import os
from dotenv import load_dotenv
load_dotenv()

import openai
import streamlit as st

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def main():
    st.title("PDF to Text")
    st.header("Upload a PDF file and ask any question about it.")

    pdf = st.file_uploader("Upload a PDF file", type=["pdf"], accept_multiple_files=False)
    return pdf


if __name__ == "__main__":
    main()