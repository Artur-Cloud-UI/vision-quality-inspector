import os
import torch
import pytorch_lightning as pl

os.environ["TRUST_REMOTE_CODE"] = "1"

# Blokada losowości dla w 100% powtarzalnych wyników (stabilny F1-score)
pl.seed_everything(42, workers=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore

def main():
    ckpt_path = "results/Patchcore/metal_nut/latest/weights/lightning/model.ckpt"
    
    # Konfiguracja folderu testowego
    datamodule = Folder(
        name="custom_dataset",
        root=".",
        normal_dir="data/train/good",
        normal_test_dir="data/test/good",
        abnormal_dir="data/test"
    )

    print("[INFO] Wczytywanie modelu...")
    model = Patchcore.load_from_checkpoint(checkpoint_path=ckpt_path)
    
    engine = Engine()

    print("[INFO] Uruchamianie testów ze stałą powtarzalnością...")
    results = engine.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=ckpt_path
    )

    print("\n[WYNIKI TESTU]:")
    print(results)

if __name__ == "__main__":
    main()