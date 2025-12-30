# Hybrid Movie Recommendation Engine

A high-performance backend microservice that combines Collaborative Filtering (SVD) and Content-Based Filtering (GenAI/BERT) to provide personalized movie recommendations.

Designed to solve the "Cold Start" problem by falling back to semantic search when user history is unavailable.

## Key Features

* **Hybrid Architecture:** intelligently switches strategies based on data availability.
    * **Engine A (SVD):** Uses Matrix Factorization to recommend movies based on historical user rating patterns (high accuracy).
    * **Engine B (GenAI):** Uses Vector Embeddings (BERT) to find movies based on semantic plot similarity (high coverage).
* **Scalable API:** Built with FastAPI for asynchronous, high-performance request handling.
* **Semantic Search:** Users can search for vague concepts (e.g., "movies about space wars") and get accurate results even if the keywords don't match the title.

## Tech Stack

* **Language:** Python 3.10
* **Framework:** FastAPI, Uvicorn
* **Machine Learning:** Scikit-Learn (TruncatedSVD), NumPy, Pandas
* **GenAI / NLP:** Sentence-Transformers (Hugging Face all-MiniLM-L6-v2)

## Setup & Installation

### 1. Clone the Repository

    git clone https://github.com/Koushik-Pula/Movie_recommender_system.git
    cd Movie_recommender_system

### 2. Create Virtual Environment

    python -m venv venv
    .\venv\Scripts\activate

### 3. Install Dependencies

    pip install -r requirements.txt

### 4. Data Setup
1.  Download the MovieLens Small Dataset from GroupLens.
2.  Extract the zip file.
3.  Place 'movies.csv' and 'ratings.csv' inside a folder named 'data/' in the root directory.

### 5. Train the Models
Since the model files are large, they are not stored in the repo. You must generate them locally (takes ~30 seconds).
1.  Open 'train_model.ipynb' in VS Code or Jupyter Notebook.
2.  Run all cells.
3.  This will create a 'models/' directory containing the trained SVD matrix and Vector Embeddings.

## Usage

**Start the API Server:**

    python main.py

**Access the UI:**
Open your browser and navigate to:
http://localhost:8000/docs

This opens the Swagger UI, where you can test the API interactively.

## API Endpoints

### GET /recommend/hybrid/{query}
The main endpoint for recommendations.

**Logic Flow:**
1.  Receives a query (e.g., "Toy Story" or "Scary ghost movie").
2.  **Check 1:** Does the movie exist in the database?
    * **Yes:** Trigger SVD Engine (Collaborative Filtering). Returns movies liked by similar users.
    * **No:** Trigger GenAI Engine (Content-Based). Converts query to a Vector Embedding and finds semantically similar movies.

**Example Response (SVD):**

    {
      "strategy": "Collaborative Filtering (SVD)",
      "data": [
        {"title": "Aladdin (1992)", "score": 0.98},
        {"title": "Monsters, Inc. (2001)", "score": 0.96}
      ]
    }

**Example Response (GenAI):**

    {
      "strategy": "Content-Based (GenAI)",
      "data": [
        {"title": "The Conjuring", "score": 0.82},
        {"title": "Insidious", "score": 0.79}
      ]
    }