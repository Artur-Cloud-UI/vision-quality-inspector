import os
# Musimy ustawić tę flagę przed importem inferencera, aby ominąć blokadę bezpieczeństwa PyTorcha
os.environ["TRUST_REMOTE_CODE"] = "1"

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import io
from PIL import Image
from pathlib import Path
import traceback

from anomalib.deploy import TorchInferencer

app = FastAPI(title="Vision Quality Inspector API", version="1.0")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "results" / "weights" / "torch" / "model.pt"

inferencer = None

@app.on_event("startup")
def load_model():
    global inferencer
    print(f"\n[STARTUP] Sprawdzam ścieżkę do modelu: {MODEL_PATH}")
    print(f"[STARTUP] Czy plik modelu istnieje? {MODEL_PATH.exists()}")
    
    if MODEL_PATH.exists():
        try:
            print("[STARTUP] Próbuję załadować TorchInferencer...")
            inferencer = TorchInferencer(path=str(MODEL_PATH))
            print("[STARTUP] SUKCES! Model został poprawnie załadowany i jest gotowy.")
        except Exception as e:
            print(f"[STARTUP] BŁĄD podczas inicjalizacji TorchInferencer: {e}")
            traceback.print_exc()
    else:
        print(f"[STARTUP] KATASTROFA: Nie znaleziono pliku .pt w lokalizacji: {MODEL_PATH}")

@app.get("/")
def home():
    return {"message": "Vision Quality Inspector API is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if inferencer is None:
        return JSONResponse(
            content={"error": "Model nie jest załadowany! Sprawdź terminal przy starcie serwera."},
            status_code=500
        )
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
        
        # Uruchomienie predykcji
        prediction = inferencer.predict(image=image_np)
        
        # Bezpieczne wyciągnięcie wyników
        anomaly_score = float(getattr(prediction, "pred_score", 0.0))
        is_anomaly = bool(getattr(prediction, "pred_label", False))
        
        print(f"-> Plik: {file.filename} | Score: {anomaly_score} | Is Defect: {is_anomaly}")

        return JSONResponse(
            content={
                "filename": file.filename,
                "status": "DEFECT" if is_anomaly else "OK",
                "anomaly_score": anomaly_score,
                "message": "Image processed successfully with TorchInferencer."
            },
            status_code=200
        )

    except Exception as e:
        print(f"Błąd podczas predykcji: {e}")
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=400)