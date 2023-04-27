from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from internal_project import *
from query_file import *
from delete_vectors import *
from flask_cors import CORS

save_path = 'C:/Users/SDADMIN23/Desktop/files/'
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route("/")
@app.route("/connection")
def home():
    return "Connection Established on port 5001"


@app.route("/readfile", methods=['POST', 'GET'])
def readfile():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        f_path = os.path.join(save_path + filename)
        f.save(f_path)
        open_api_key, pinecone_api_key, pinecone_api_env, index_description = get_file(
            f_path, filename)

        # Open the file in write mode (this will overwrite the file)
        with open('credentials.txt', 'w') as file:
            # Write new content to the file
            file.write(open_api_key + '\n' + pinecone_api_key + '\n' +
                       pinecone_api_env + '\n' + 'langchain-internal-project' + '\n' + filename)

        data = jsonify(index_name='langchain-internal-project',
                       namespace=filename)
        status = 200

        return data, status,  {'Access-Control-Allow-Origin': '*'}


@app.route("/queryfile", methods=['POST', 'GET'])
def queryfile():
    if request.method == 'POST':
        query = request.form['query']
        f = open("credentials.txt", "r")
        lines = f.read().splitlines()
        open_api_key = lines[0]
        pinecone_api_key = lines[1]
        pinecone_api_env = lines[2]
        index_name = lines[3]
        namespace = lines[4]
        f.close()

        query_result = get_query_results(
            open_api_key, pinecone_api_key, pinecone_api_env, index_name, namespace, query)

        query_result = ''.join(query_result.splitlines())
        query_result.encode("utf-8")
        data = jsonify(query_result)
        status = 200
        return data, status,  {'Access-Control-Allow-Origin': '*'}


@app.route("/deleteVectors", methods=['GET'])
def deleteVectors():
    if request.method == 'GET':
        f = open("credentials.txt", "r")
        lines = f.read().splitlines()
        pinecone_api_key = lines[1]
        pinecone_api_env = lines[2]
        index_name = lines[3]
        f.close()
        delete_index_vectors(pinecone_api_key, pinecone_api_env, index_name)
        status = 200
        return 'Vectors Deleted', status,  {'Access-Control-Allow-Origin': '*'}


if __name__ == '__main__':
    app.run(debug=True, port=5001)
