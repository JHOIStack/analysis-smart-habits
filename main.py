# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from main_supervised import classify_profiles
from main_unsupervised import cluster_profiles

app = FastAPI(title="Smart Habits ML API")

# Permitir CORS para que Express pueda consumir la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Smart Habits ML API is running"}

@app.get("/ml/supervised")
def get_supervised():
    try:
        result = classify_profiles()
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/ml/unsupervised")
def get_unsupervised():
    try:
        result = cluster_profiles()
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
