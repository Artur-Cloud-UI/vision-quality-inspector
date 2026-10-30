import os
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine

def find_latest_checkpoint(results_dir="results"):
    """Automatycznie szuka pliku wagi (.ckpt) wygenerowanego podczas treningu."""
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith(".ckpt"):
                return os.path.join(root, file)
    return None

def main():
    print("[INFO] Szukam wytrenowanego modelu w folderze 'results'...")
    ckpt_path = find_latest_checkpoint("results")
    
    if not ckpt_path:
        print("[BŁĄD] Nie znaleziono pliku .ckpt! Upewnij się, że najpierw pomyślnie uruchomiłeś train.py.")
        return
    
    print(f"[INFO] Znaleziono model (checkpoint): {ckpt_path}")

    print("[INFO] Konfiguracja modułu danych...")
    datamodule = Folder(
        name="metal_nut",
        root="data",
        normal_dir="train/good",
        abnormal_dir="test",
        train_batch_size=16,
        eval_batch_size=16,
    )

    print("[INFO] Inicjalizacja modelu Patchcore...")
    model = Patchcore()

    print("[INFO] Inicjalizacja silnika (Engine)...")
    engine = Engine(
        default_root_dir="results"
    )

    print("[INFO] Uruchamianie testów z załadowaną pamięcią modelu...")
    # Przekazujemy ścieżkę do checkpointu, żeby model wiedział, co ma w pamięci
    engine.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    
    print("\n[SUKCES] Ewaluacja zakończona!")
    print("Sprawdź folder 'results/' – znajdziesz tam wygenerowane metryki oraz mapy ciepła (heatmapy) wad.")

if __name__ == "__main__":
    main()