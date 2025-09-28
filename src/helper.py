from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document


# Extract text from pdf
def load_pdf(data):
    loader=DirectoryLoader(
        data,
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )
    
    documents=loader.load()
    return documents

# Preprocessing
def filter_docs(text:List[Document])->List[Document]:
    reduced_docs:List[Document]=[]
    for i in text:
        source=i.metadata.get('source')
        reduced_docs.append(
            Document(
                page_content=i.page_content,
                metadata={"source":source}
            )
        )
    return reduced_docs


# Chunking
def chunking(docs):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    chunks=text_splitter.split_documents(docs)
    return chunks
    
    
# Calling embedding model
def embedding_model():
    model_name='sentence-transformers/all-MiniLM-L6-v2'
    embeddings=HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings

