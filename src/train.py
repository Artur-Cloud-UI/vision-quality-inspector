import os
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore


def main():
  print("[INFO] Konfiguracja danych przemysłowych...")

  datamodule = Folder(
      name="metal_nut",
      root="data",  # folder główny zawierający 'train/good'
      normal_dir="train/good",
      abnormal_dir="test",
      train_batch_size=4,  # Bezpieczne dla 2 GB VRAM
      eval_batch_size=8,
  )

  print(
      "[INFO] Inicjalizacja modelu PatchCore (Memory Bank Anomaly"
      " Detection)..."
  )
  # Zwiększamy współczynnik zapamiętywania z 0.1 na 0.3
  model = Patchcore(coreset_sampling_ratio=0.3, backbone="wide_resnet50_2")

  print("[INFO] Konfiguracja silnika treningowego (Engine)...")
  engine = Engine(default_root_dir="results")

  print(
      "[INFO] Rozpoczynamy proces budowania bazy wiedzy o poprawnej"
      " nakrętce..."
  )
  engine.fit(model=model, datamodule=datamodule)

  print(
      "\n[SUKCES] Trening zakończony! Nowy model został zapisany w folderze"
      " 'results'."
  )


if __name__ == "__main__":
  main()