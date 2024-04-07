'''
requirements.txt file contents:

langchain==0.0.154
PyPDF2==3.0.1
python-dotenv==1.0.0
streamlit==1.18.1
faiss-cpu==1.7.4
streamlit-extras
'''

import io
from io import BytesIO
import base64

import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

def show_pdf(uploaded_file):
    # with open(file_path,"rb") as f:
    with io.BytesIO() as buffer:
        buffer.write(uploaded_file.read())
        buffer.seek(0)
        base64_pdf = base64.b64encode(buffer.read()).decode('utf-8')

        # pdf_display = F'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

# Sidebar filters
def get_page():
    page = st.sidebar.text_input(label='PDF Page') 
    if str(page).strip() not in ['all','']:
        try:
            page = int(page)
            page = page - 1
        except:
            st.write('Enter a page number (starting from 1), or "all"')
    return str(page)

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model

    ''')
    add_vertical_space(5)
    # st.write('Made with ❤️ by [Prompt Engineer](https://youtube.com/@engineerprompt)')

load_dotenv()

def main():
    st.header("Chat with PDF 💬")

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    # st.write(pdf)
    if pdf is not None:

        show_pdf(pdf)

        pdf_reader = PdfReader(pdf)
        
        text = ""
        page_no = get_page()
        show_text=False
        if st.sidebar.checkbox('Show text'):
            show_text = True
        
        if page_no:
            # st.write(f'Page number: {page_no}')
            if page_no!='all':
                text = pdf_reader.pages[int(page_no)].extract_text()
            elif page_no=='all':
                st.write(f'Number of pages: {len(pdf_reader.pages)}')
                for page in pdf_reader.pages:
                    text += page.extract_text()

            # if st.sidebar.button('Extract'):
            if show_text:
                st.write(text)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
                )
            chunks = text_splitter.split_text(text=str(text))

            # # embeddings
            store_name = pdf.name[:-4]
            # st.write(f'PDF name: {store_name}')
            # st.write(chunks)

            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
                # st.write('Embeddings Loaded from the Disk')s
            else:
                embeddings = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)

            # embeddings = OpenAIEmbeddings()
            # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

            # Accept user questions/query
            query = st.text_input("Ask questions about your PDF file:")
            # st.write(query)

            if query:
                if st.button('Chat!'):
                    docs = VectorStore.similarity_search(query=query, k=5)
                    llm = OpenAI()
                    chain = load_qa_chain(llm=llm, chain_type="stuff")
                    with get_openai_callback() as cb:
                        response = chain.run(input_documents=docs, question=query)
                        print(cb)
                    st.write(response)
        elif not page_no:
            pass

if __name__ == '__main__':
    main()