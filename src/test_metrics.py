import os
os.environ["TRUST_REMOTE_CODE"] = "1"

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore

def main():
    # Ścieżka do Twojego checkpointu
    ckpt_path = "results/Patchcore/metal_nut/latest/weights/lightning/model.ckpt"
    
    # Konfiguracja folderu z uwzględnieniem normal_dir (treningowe dobre) oraz testowych
    datamodule = Folder(
        name="custom_dataset",
        root=".",                    # Aktualny folder projektu
        normal_dir="data/train/good",     # Ścieżka do dobrych obrazów z treningu
        normal_test_dir="data/test/good", # Ścieżka do dobrych obrazów testowych
        abnormal_dir="data/test"          # Folder testowy zawierający podfoldery z wadami (np. scratch)
    )

    print("[INFO] Wczytywanie modelu...")
    model = Patchcore.load_from_checkpoint(checkpoint_path=ckpt_path)
    
    engine = Engine()

    print("[INFO] Uruchamianie testów i obliczanie metryk (AUROC, F1)...")
    results = engine.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=ckpt_path
    )

    print("\n[WYNIKI TESTU]:")
    print(results)

if __name__ == "__main__":
    main()