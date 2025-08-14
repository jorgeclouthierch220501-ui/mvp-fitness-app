# MVP Fitness App (FastAPI)

API con:
- Conteo calórico (registro por día)
- Planes de dieta y entrenamiento generados por IA (OpenAI)

## Correr en Codespaces
1) Abre un Codespace: Code → Codespaces → Create codespace on main
2) Crea un archivo `.env` (no lo subas a git) con:
   OPENAI_API_KEY=TU_API_KEY
3) En terminal:
   pip install -r requirements.txt
   uvicorn app:app --host 0.0.0.0 --port 8000
4) Abre la URL del puerto 8000 → `/docs` para probar.

## Deploy en Render
Conecta el repo, Build: `pip install -r requirements.txt`,
Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`,
y agrega la variable `OPENAI_API_KEY`.
