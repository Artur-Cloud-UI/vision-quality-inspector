import os
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine

def main():
    print("[INFO] Konfiguracja danych przemysłowych...")
    
    datamodule = Folder(
        name="metal_nut",
        root="data",
        normal_dir="train/good",
        abnormal_dir="test",
        train_batch_size=16,
        eval_batch_size=16,
    )

    print("[INFO] Inicjalizacja modelu PatchCore (Memory Bank Anomaly Detection)...")
    model = Patchcore()

    print("[INFO] Konfiguracja silnika treningowego (Engine)...")
    
    engine = Engine(
        default_root_dir="results"
    )

    print("[INFO] Rozpoczynamy proces budowania bazy wiedzy o poprawnej nakrętce...")
    engine.fit(model=model, datamodule=datamodule)
    
    print("\n[SUKCES] Trening zakończony! Model poprawnie przyswoił wzorzec i zapisał wyniki w folderze 'results'.")

if __name__ == "__main__":
    main()