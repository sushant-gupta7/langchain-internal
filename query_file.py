import os
from llama_index import download_loader
from langchain.llms import OpenAI
import openai
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from pathlib import Path
import pinecone
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.docstore.document import Document
from tqdm.auto import tqdm
import tiktoken
from pathlib import Path
from gpt_index import download_loader
from sentence_transformers import SentenceTransformer
from transformers import pipeline


def initialize_open_ai(OPENAI_API_KEY):
    openai.api_key = OPENAI_API_KEY


def initialize_pinecone(PINECONE_API_KEY, PINECONE_API_ENV, index_name):

    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_API_ENV # find next to API key in console
    )

    index = pinecone.Index(index_name)
    return index

def retrieve(index, query, namespace):

    embed_model = "text-embedding-ada-002"

    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )

    # retrieve from Pinecone
    xq = res['data'][0]['embedding']

    # get relevant contexts
    res = index.query(xq, top_k=3, include_metadata=True, namespace=namespace)
    contexts = [
        x['metadata']['text'] for x in res['matches']
    ]

    # build our prompt with the retrieved contexts included
    prompt_start = (
        "Answer the question based on the context below.\n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )
    # append contexts until hitting limit
    
    prompt = (prompt_start + "\n\n---\n\n".join(contexts) + prompt_end)

    return prompt

def complete(prompt):
    # query text-davinci-003
    res = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return res['choices'][0]['text'].strip()

def get_query_results(open_api_key, pinecone_api_key, pinecone_api_env, index_name, namespace, query):
    initialize_open_ai(open_api_key)
    index = initialize_pinecone(pinecone_api_key, pinecone_api_env, index_name)
    query_with_contexts = retrieve(index, query, namespace)
    data = complete(query_with_contexts)

    return data