import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Hybrid Movie Recommender")

print("Loading models...")
try:
    with open('models/movies.pkl', 'rb') as f:
        movies = pickle.load(f)
    
    with open('models/svd_matrix.pkl', 'rb') as f:
        svd_matrix = pickle.load(f)
        
    with open('models/genai_embeddings.pkl', 'rb') as f:
        genai_embeddings = pickle.load(f)
        
    with open('models/movie_map.pkl', 'rb') as f:
        movie_index_map = pickle.load(f)

    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Models Loaded Successfully!")
except Exception as e:
    print(f"ERROR: Could not load models. Did you run the notebook? {e}")


def get_svd_recommendations(movie_title, k=5):
    """Engine A: Collaborative Filtering"""
   
    match = movies[movies['title'].str.contains(movie_title, case=False)]
    if match.empty: return None
    
    movie_id = match.iloc[0]['movieId']
    
    if movie_id not in movie_index_map: return None
    
    idx = movie_index_map.index(movie_id)
    sim_scores = svd_matrix[idx]
    
    top_indices = sim_scores.argsort()[::-1][:k+1]
    results = []
    for i in top_indices:
        rec_id = movie_index_map[i]
        if rec_id != movie_id:
            title = movies[movies['movieId'] == rec_id]['title'].values[0]
            results.append({"title": title, "score": float(sim_scores[i]), "type": "SVD"})
            
    return results[:k]

def get_genai_recommendations(text_query, k=5):
    """Engine B: Content-Based (GenAI)"""
  
    query_vec = bert_model.encode([text_query])
    
    sim_scores = cosine_similarity(query_vec, genai_embeddings).flatten()
    
    top_indices = sim_scores.argsort()[::-1][:k]
    results = []
    for i in top_indices:
        title = movies.iloc[i]['title']
        results.append({"title": title, "score": float(sim_scores[i]), "type": "GenAI"})
        
    return results



@app.get("/")
def home():
    return {"status": "active", "models_loaded": True}

@app.get("/recommend/hybrid/{query}")
def hybrid_recommend(query: str):
    """
    The Smart Endpoint.
    1. Tries to find the movie and use SVD (User patterns).
    2. If that fails, treats the input as a search phrase and uses GenAI.
    """
    print(f"Received query: {query}")
    
    recommendations = get_svd_recommendations(query)
    
    if recommendations:
        return {
            "strategy": "Collaborative Filtering (SVD)",
            "note": "Based on what other users liked.",
            "data": recommendations
        }
    
    print("SVD failed/Movie not found. Switching to GenAI Search...")
    recommendations = get_genai_recommendations(query)
    
    return {
        "strategy": "Content-Based (GenAI)",
        "note": "Based on semantic similarity.",
        "data": recommendations
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)