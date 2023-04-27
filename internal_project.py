#!/usr/bin/env python
# coding: utf-8

import os
from llama_index import download_loader
from langchain.llms import OpenAI
import openai
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
from uuid import uuid4
import datetime
from time import sleep


def load_file_data(file_path):
    PDFReader = download_loader("PDFReader")
    loader = PDFReader()
    loaded_data = loader.load_data(file=Path(file_path))
    return loaded_data

# create the length function used by the RecursiveCharacterTextSplitter


def tiktoken_len(text):
    # use cl100k_base tokenizer for gpt-3.5-turbo and gpt-4
    tokenizer = tiktoken.get_encoding('cl100k_base')
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


def split_text():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=tiktoken_len, separators=['\n\n', '\n', ' ', ''])
    return text_splitter


def create_chunks(data, text_splitter):
    chunks = []
    for idx, record in enumerate(tqdm(data)):
        texts = text_splitter.split_text(record.text)
        chunks.extend([{'id': str(uuid4()), 'text': texts[i], 'chunk': i}
                      for i in range(len(texts))])
    return chunks


def initiaize_keys():
    OPENAI_KEY = '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'
    PINECONE_KEY = 'd26f38f0-4937-4f00-8591-a766922f09e8'
    PINECONE_ENV = 'us-west4-gcp'
    return OPENAI_KEY, PINECONE_KEY, PINECONE_ENV


def initialize_open_ai(OPENAI_API_KEY):
    openai.api_key = OPENAI_API_KEY


def initialize_pinecone(PINECONE_API_KEY, PINECONE_API_ENV):
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_API_ENV  # find next to API key in console
    )
    # check if 'openai' index already exists (only create index if not)
    if 'langchain-internal-project' not in pinecone.list_indexes():
        pinecone.create_index('langchain-internal-project', dimension=1536)
    # connect to index
    index = pinecone.Index('langchain-internal-project')
    return index


def get_vectors(chunks):

    embed_model = "text-embedding-ada-002"

    batch_size = 100  # how many embeddings we create and insert at once

    for i in tqdm(range(0, len(chunks), batch_size)):
        # find end of batch
        i_end = min(len(chunks), i+batch_size)
        meta_batch = chunks[i:i_end]
        # get ids
        ids_batch = [x['id'] for x in meta_batch]
        # get texts to encode
        texts = [x['text'] for x in meta_batch]
        # create embeddings (try-except added to avoid RateLimitError)
        try:
            res = openai.Embedding.create(input=texts, engine=embed_model)
        except:
            done = False
            while not done:
                sleep(5)
                try:
                    res = openai.Embedding.create(
                        input=texts, engine=embed_model)
                    done = True
                except:
                    pass
        embeds = [record['embedding'] for record in res['data']]
        # cleanup metadata
        meta_batch = [{'text': x['text'], 'chunk': x['chunk']}
                      for x in meta_batch]
        to_upsert = list(zip(ids_batch, embeds, meta_batch))
        return to_upsert


def add_vectors_to_pinecone(index, to_upsert, filename):
    # upsert to Pinecone
    index.upsert(vectors=to_upsert, namespace=filename)


def get_file(file_p, filename):
    data = load_file_data(file_p)
    text_splitter = split_text()
    chunks = create_chunks(data, text_splitter)
    OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_API_ENV = initiaize_keys()
    initialize_open_ai(OPENAI_API_KEY)
    index = initialize_pinecone(PINECONE_API_KEY, PINECONE_API_ENV)
    whoami_data = pinecone.whoami()
    index_description = index.describe_index_stats()
    if(filename in index_description.namespaces):
        index.delete(deleteAll='true', namespace=filename)
    to_upsert = get_vectors(chunks)
    add_vectors_to_pinecone(index, to_upsert, filename)
    return OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_API_ENV, 'langchain-internal-project'
