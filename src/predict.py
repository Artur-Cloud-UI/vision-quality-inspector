import os
os.environ["TRUST_REMOTE_CODE"] = "1"

from anomalib.deploy import TorchInferencer
from anomalib.engine import Engine
from anomalib.models import Patchcore
from PIL import Image
import numpy as np
import torch

# ==========================================
# ⚙️ TUTAJ WPISZ SWOJĄ DOKŁADNĄ ŚCIEŻKĘ DO TESTÓW:
TEST_DIR = "data/test" 
# ==========================================

def main():
    print("[INFO] Szukam wytrenowanego modelu w folderze 'results'...")
    
    ckpt_path = None
    for root, _, files in os.walk("results"):
        for file in files:
            if file == "model.ckpt":
                ckpt_path = os.path.join(root, file)
                break
        if ckpt_path:
            break

    if not ckpt_path:
        print("[BŁĄD] Nie znaleziono pliku .ckpt!")
        return

    print(f"[INFO] Znaleziono model: {ckpt_path}")

    # Eksport do .pt (jeśli jeszcze nie zrobiony)
    exported_pt_path = "results/exported/weights/torch/model.pt"
    if not os.path.exists(exported_pt_path):
        print("[INFO] Eksportowanie modelu do formatu .pt...")
        try:
            model = Patchcore.load_from_checkpoint(checkpoint_path=ckpt_path)
            engine = Engine()
            engine.export(model=model, export_type="torch", export_root="results/exported", ckpt_path=ckpt_path)
        except Exception as e:
            print(f"[INFO] Pomijam eksport: {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Używane urządzenie: {device.upper()}")

    print(f"[INFO] Inicjalizacja inferencera...")
    inferencer = TorchInferencer(path=exported_pt_path, device=device)

    if not os.path.exists(TEST_DIR):
        print(f"[BŁĄD] Podana ścieżka testowa nie istnieje: {TEST_DIR}")
        return

    base_output_dir = "results/all_heatmaps"
    os.makedirs(base_output_dir, exist_ok=True)

    print(f"[INFO] Przetwarzanie plików z zachowaniem struktury folderów z: {TEST_DIR} ...")
    
    count = 0
    for root, dirs, files in os.walk(TEST_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(root, file)
                
                # KLUCZOWE: Obliczamy ścieżkę względną, np. "scratch" lub "good"
                rel_path = os.path.relpath(root, TEST_DIR)
                
                # Tworzymy taki sam podfolder w wynikach
                target_dir = os.path.join(base_output_dir, rel_path)
                os.makedirs(target_dir, exist_ok=True)
                
                # Ścieżka zapisu konkretnej heatmapy w odpowiednim podfolderze
                save_path = os.path.join(target_dir, f"heatmap_{file}")
                
                try:
                    predictions = inferencer.predict(image=img_path)
                    
                    if hasattr(predictions, "anomaly_map"):
                        anomaly_map = predictions.anomaly_map
                    elif isinstance(predictions, dict) and "anomaly_map" in predictions:
                        anomaly_map = predictions["anomaly_map"]
                    else:
                        anomaly_map = getattr(predictions, "heat_map", None)

                    if anomaly_map is not None:
                        if hasattr(anomaly_map, "detach"):
                            anomaly_map = anomaly_map.detach().cpu().numpy()
                        
                        anomaly_map = np.squeeze(anomaly_map)
                        
                        if anomaly_map.max() > anomaly_map.min():
                            anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
                        anomaly_map = (anomaly_map * 255).astype(np.uint8)
                        
                        Image.fromarray(anomaly_map).save(save_path)
                        count += 1
                except Exception as ex:
                    print(f"[BŁĄD] Plik {file}: {ex}")

    print(f"\n[SUKCES] Gotowe! Zapisano {count} heatmap z zachowaniem podfolderów w: {base_output_dir}")

if __name__ == "__main__":
    main()