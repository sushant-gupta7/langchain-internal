import openai
import pinecone


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
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content":"You are an assistant that only provides relevant answers."},
                  {'role': 'user', 'content':  "Answer me only if the file below the --- is relevant to the question. If not relevant say so and provide an answer beyond the uploaded file. If relevant, answer in detail" + prompt}],
        max_tokens=1024,
        n=1,
        temperature=0,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
    )
    # Get the response text from the API response
    response_text = res['choices'][0]['message']['content']

    return response_text

def get_query_results(open_api_key, pinecone_api_key, pinecone_api_env, index_name, namespace, query):
    initialize_open_ai(open_api_key)
    index = initialize_pinecone(pinecone_api_key, pinecone_api_env, index_name)
    query_with_contexts = retrieve(index, query, namespace)
    data = complete(query_with_contexts)
    return data