import pinecone

def initialize_pinecone(PINECONE_API_KEY, PINECONE_API_ENV, index_name):

    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_API_ENV # find next to API key in console
    )

    index = pinecone.Index(index_name)
    return index

def delete_index_vectors(pinecone_api_key, pinecone_api_env, index_name):
    index = initialize_pinecone(pinecone_api_key, pinecone_api_env, index_name)
    index_description = index.describe_index_stats()
    print('index_description', index_description)
    print('index_description.namespaces', index_description.namespaces)
    for key in index_description.namespaces.keys():
        index.delete(deleteAll='true', namespace=key)
    return
    