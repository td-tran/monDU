from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.search_engine import SearchEngine

app = FastAPI(title="Formation Search API", version="0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"]
)

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "du.csv"
engine = SearchEngine(CSV_PATH)


@app.get("/health")
def health_check():
    return {"status": "ok", "index_size": len(engine.titles)}


@app.get("/search")
def search(
    query: str = Query(..., min_length=1),
    k: int = Query(10, ge=1, le=50),
    reranker: str = Query("hybrid"),
):
    results = engine.search(query, top_k=k, reranker=reranker)
    return {"query": query, "reranker": reranker, "count": len(results), "results": results}


@app.post("/reindex")
def reindex():
    engine.force_reindex()
    return {"status": "reindexed", "index_size": len(engine.titles)}
