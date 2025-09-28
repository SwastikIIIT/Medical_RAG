from dotenv import load_dotenv
import os
from src.helper import load_pdf, filter_docs, chunking, embedding_model
from pinecone import Pinecone
from pinecone import ServerlessSpec 
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
google_key=os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')

os.environ['PINECONE_API_KEY']=PINECONE_API_KEY
os.environ['GOOGLE_APPLICATION_CREDENTIALS']=google_key

text=load_pdf(data='data/')
filtered_document=filter_docs(text)
chunked_data=chunking(filtered_document)

embed_model=embedding_model()

pc=Pinecone(api_key=PINECONE_API_KEY)
indexName='medical-bot'

if not pc.has_index(indexName):
    pc.create_index(
        name=indexName,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws',region='us-east-1')
    )
    
index=pc.Index(indexName)


docsearch=PineconeVectorStore.from_documents(
    documents=chunked_data,
    embedding=embed_model,
    index_name=indexName
)