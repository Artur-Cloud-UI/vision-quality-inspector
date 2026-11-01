import torch

ckpt_path = "results/patchcore/metal_nut/v2/weights/lightning/model.ckpt"  # upewnij się, że ścieżka jest poprawna

try:
  checkpoint = torch.load(
      ckpt_path, map_location="cpu", weights_only=False
  )

  print("=== PREPROCESSOR CONFIG ===")
  if "anomalib_pre_processor_config" in checkpoint:
    pp_config = checkpoint["anomalib_pre_processor_config"]
    if isinstance(pp_config, dict):
      for k, v in pp_config.items():
        print(f"{k}: {v}")
    else:
      print(pp_config)
  else:
    print("Brak klucza anomalib_pre_processor_config.")

except Exception as e:
  print(f"Błąd: {e}")