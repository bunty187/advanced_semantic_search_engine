from flask import Flask, request, render_template
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Initialize the ChromaDB client
CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "multi-qa-mpnet-base-dot-v1"
COLLECTION_NAME = "movie_subtitle_collection"

client = PersistentClient(path=CHROMA_DATA_PATH)
embedding_model = SentenceTransformer(EMBED_MODEL)

# Load the collection
collection = client.get_collection(name=COLLECTION_NAME)

# @app.route('/')
# def home():
#     # Render the home page with the search form
#     return render_template('index.html')

@app.route('/', methods=['GET','POST'])
def search():
    if request.method == 'POST':
        # Get the user query from the form
        user_query = request.form['query']
        # Generate embeddings for the user query using the embedding model
        user_query_embedding = embedding_model.encode([user_query], convert_to_tensor=True)
        # Perform the search using ChromaDB
        results = perform_search(user_query_embedding)
        # Render the search results page
        return render_template('index.html', query=user_query, results=results)
    return render_template('index.html')  # Return form on GET request


def perform_search(query_embedding):
    # Perform a ChromaDB query and return the top 10 results
    query_results = collection.query(
        query_embeddings=query_embedding.tolist(),  # Convert tensor to list
        n_results=10
    )
    # Extract the required information from the query results
    ids = query_results['ids']
    distances = query_results['distances']
    metadatas = query_results['metadatas']
    documents = query_results['documents']

    # Combine the extracted information into a list of dictionaries
    results = [
        {
            'id': id_item,
            'distance': distance_item,
            'metadata': metadata_item,
            'document': document_item
        }
        for id_item, distance_item, metadata_item, document_item in zip(ids, distances, metadatas, documents)
    ]
    return results


if __name__ == '__main__':
    app.run(debug=True)
