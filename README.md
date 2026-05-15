# monDU - Moteur de recherche de formations

## Architecture
- `backend/` : API Python FastAPI pour la recherche
- `frontend/` : interface JavaScript statique pour tester le backend
- `data/du.csv` : liste des formations à rechercher

## Fonctionnement
1. Backend charge `data/du.csv` et calcule :
   - recherche lexicale avec TF-IDF
   - recherche sémantique avec `sentence-transformers`
2. Le backend fusionne les meilleurs candidats des deux méthodes
3. Un reranker trie les résultats sur un score combiné

## Lancer le backend
1. Ouvrir un terminal pour aller au répertoir monDU
2. Activer l'environnement virtuel (si Windows) :
   ```powershell
   venv\Scripts\activate
   ```
3. Installer les dépendances :
   ```powershell
   pip install -r backend/requirements.txt
   ```
4. Lancer le serveur :
   ```powershell
   python -m uvicorn backend.main:app --reload
   ```
5. Vérifier l'API : `http://127.0.0.1:8000/health`

## Configuration de l'environnement
1. Créer l'environnement virtuel :
   ```powershell
   python -m venv venv
   ```
2. L'activer :
   ```powershell
   venv\Scripts\activate
   ```
3. Installer les dépendances :
   ```powershell
   pip install -r backend/requirements.txt
   ```

## Configuration des données
Les développeurs doivent ajouter `data/du.csv` localement.
Ce fichier n'est pas versionné pour des raisons de confidentialité.

## Endpoint de réindexation
- `POST /reindex` : force la reconstruction de l'index si tu veux forcer la mise à jour

## Reranker disponibles
- `none` : pas de reranker, fusion simple des résultats lexicaux et sémantiques
- `hybrid` : reranker intégré (TF-IDF + sémantique + overlap) 
- `cross-ms-marco` : CrossEncoder `cross-encoder/ms-marco-MiniLM-L6-v2`

## lancer le backend depuis le répertoire monDU
- python -m pip install -r backend/requirements.txt
- python -m uvicorn backend.main:app --reload
- vérifier http://127.0.0.1:8000/health

## Lancer le frontend
- depuis \monDU\frontend
- python -m http.server 5500
- http://127.0.0.1:5500/index.html 
- Le frontend fera des requêtes vers `http://127.0.0.1:8000/search?query=...&reranker=hybrid`

## Endpoints
- `GET /health` : statut du service
- `GET /search?query=<texte>&k=<nombre>` : recherche de formations
