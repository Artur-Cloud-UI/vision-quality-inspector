from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import Patchcore

# 1. Ścieżka do Twojego pliku .ckpt po zakończeniu treningu
CHECKPOINT_PATH = (
    "results/Patchcore/metal_nut/v5/weights/lightning/model.ckpt"
)

print(f"[INFO] Wczytywanie wytrenowanego modelu z: {CHECKPOINT_PATH}...")

# 2. Inicjalizacja modelu i wczytanie wag
model = Patchcore.load_from_checkpoint(checkpoint_path=CHECKPOINT_PATH)

# 3. Utworzenie silnika i eksport do model.pt
engine = Engine()
engine.export(
    model=model, export_type=ExportType.TORCH, export_root="results/exported"
)

print(
    "\n[SUKCES] Wyeksportowano model.pt! Znajdziesz go w folderze:"
    " results/exported/weights/torch/model.pt"
)