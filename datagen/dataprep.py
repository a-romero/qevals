import os
import config
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredMarkdownLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def convert_to_text(path_array):
    doc_list = []

    for file in os.listdir(path_array):
        if file.endswith('.pdf'):
            loader = PyPDFLoader(file)
            doc_list.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'):
            loader = Docx2txtLoader(file)
            doc_list.extend(loader.load())
        elif file.endswith('.md'):
            loader = UnstructuredMarkdownLoader(file)
            doc_list.extend(loader.load())
        elif file.endswith('.txt'):
            loader = TextLoader(file)
            doc_list.extend(loader.load())
        else:
            print(f"File: {file} not supported")

    return doc_list

def preprocess(doc_list):
    separators = [' ', '.', ',', '\n', ';', ':']
    chunked_docs = []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.config['DATAGEN']['CHUNK_SIZE'],
                                                    chunk_overlap=config.config['DATAGEN']['CHUNK_OVERLAP'],
                                                    separators=separators)

    chunked_docs = text_splitter.split_documents(doc_list)

    return chunked_docs
