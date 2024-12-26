# Import necessary libraries
import streamlit as st
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.chains import RetrievalQAWithSourcesChain
import PyPDF2

# Extract texts inside PDF
def pdf_to_text(files):
  text_list = []
  source_list = []
  for file in files:
    pdf_reader = PyPDF2.PdfReader(file)
    for i in range(len(pdf_reader.pages)):
      page = pdf_reader.pages[i]
      text = page.extract_text()
      page.clear()
      text_list.append(text)
      source_list.append(file.name + "_page_" + "str(i)")
  return [text_list, source_list]

# Customize PDF Reviewer Website
st.set_page_config(layout = 'centered', page_title = "PDF Reviewer")
st.header("PDF Reviewer")
st.write("---")

# File to be upload
upload_files = st.file_uploader("Upload up to 3 PDF Documents", accept_multiple_files = True, type = ['pdf'])

# Check files upload status
if upload_files:
  if len(upload_files) > 3:
    st.warning("You can only upload up to 3 PDF Documents")
    upload_files = upload_files[:3]
  else:
    st.success(f"{len(upload_files)} document(s) ready for processing")

# When Process Button is Pressed
if st.button("Process Files"):
  if not upload_files:
    st.info("Please upload PDF Documents")
  else:
    with st.spinner("Processing Files..."):
      try:
        # Extract text and sources
        text_and_source = pdf_to_text(upload_files)
        text = text_and_source[0]
        source = text_and_source[1]

        # Extract embeddings
        embeddings = OpenAIEmbeddings(openai_api_key = st.secrets["OPENAI_API_KEY"])

        # Vector store with metadata
        vectordb = Chroma.from_texts(text, embeddings, metadatas = [{"source": s} for s in source])

        # Retrieval model
        llm = OpenAI(model_name = "gpt-3.5-turbo", openai_api_key = st.secrets["OPENAI_API_KEY"])
        retriever = vectordb.as_retriever(search_kwargs = {"k":2})
        model = RetrievalQAWithSourcesChain.from_chain_type(llm = llm, chain_type = "stuff", retriever = retriever)

        # Ask some questions
        st.header("Ask something about you uploaded")
        query = st.text_area("Enter your questions here")

        if st.button("Get Answer"):
          try:
            with st.spinner("Model is working on it...")
            result = model({"question": query}, return_only_outputs = True)
            st.subheader("Answer:")
            st.write(result["answer"])
            st.subheader("Source Pages:")
            st.write(result["sources"])
          except Exception as e:
            st.error(f"An error occured: {e}")
      except Exception as e:
        st.error(f"An error occured during processing: {e}")
