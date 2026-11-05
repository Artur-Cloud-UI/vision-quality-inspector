import os
os.environ["TRUST_REMOTE_CODE"] = "1"

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
from PIL import Image
from pathlib import Path
import traceback
from contextlib import asynccontextmanager

from anomalib.deploy import TorchInferencer

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "results" / "exported" / "weights" / "torch" / "model.pt"

inferencer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global inferencer
    print(f"\n[STARTUP] Sprawdzam ścieżkę do modelu: {MODEL_PATH}")
    if MODEL_PATH.exists():
        try:
            print("[STARTUP] Ładowanie TorchInferencer (to może chwilę potrwać na CPU)...")
            inferencer = TorchInferencer(path=str(MODEL_PATH))
            print("[STARTUP] SUKCES! Model został pomyślnie załadowany.")
        except Exception as e:
            print(f"[STARTUP] BŁĄD podczas ładowania modelu: {e}")
            traceback.print_exc()
    else:
        print(f"[STARTUP] BŁĄD: Nie znaleziono pliku modelu pod ścieżką: {MODEL_PATH}")
    
    yield
    print("[SHUTDOWN] Zamykanie aplikacji...")

# Tworzymy aplikację JEDEN raz z poprawnym lifespan
app = FastAPI(
    title="Vision Quality Inspector API", 
    version="1.0", 
    lifespan=lifespan
)

@app.get("/")
def home():
    return {"message": "Vision Quality Inspector API is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if inferencer is None:
        return JSONResponse(
            content={"error": "Model nie jest załadowany! Sprawdź terminal przy starcie serwera."},
            status_code=500,
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Dopasowanie do rozdzielczości treningu (448x448)
        image = image.resize((448, 448))

        prediction = inferencer.predict(image=image)

        anomaly_score = float(getattr(prediction, "pred_score", 0.0))
        is_anomaly = bool(getattr(prediction, "pred_label", False))

        print(f"-> Plik: {file.filename} | Score: {anomaly_score:.4f} | Is Defect: {is_anomaly}")

        return JSONResponse(
            content={
                "filename": file.filename,
                "status": "DEFECT" if is_anomaly else "OK",
                "anomaly_score": anomaly_score,
                "message": "Image processed successfully with TorchInferencer.",
            },
            status_code=200,
        )

    except Exception as e:
        print(f"Błąd podczas predykcji: {e}")
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=400)

# Umożliwia uruchomienie serwera bezpośrednio przez 'python src/api.py'
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)