from pathlib import Path
from ultralytics import YOLO


def train_model(
    data_yaml: str = "dataset.yaml",
    epochs: int = 20,
    imgsz: int = 640,
    batch_size: int = 4,
    weights: str = "yolov8n.pt",
):
    """Esegue il fine-tuning del modello YOLO per la detection delle formiche."""
    print(f"--- Inizio Addestramento Modello ({weights}) ---")

    model = YOLO(weights)

    # 1. Avvia l'addestramento
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        name="ant_detector",
        project="outputs/runs",
        optimizer="AdamW",
        lr0=0.001,
        plots=True,
    )

    # 2. Recupera la cartella esatta di salvataggio dall'oggetto results
    save_dir = Path(results.save_dir)
    best_weights_src = save_dir / "weights" / "best.pt"
    target_weights_dst = Path("models/ant_detector_best.pt")

    # 3. Sposta i pesi migliori nella cartella models/
    if best_weights_src.exists():
        target_weights_dst.parent.mkdir(parents=True, exist_ok=True)
        # Copia o rinomina
        import shutil

        shutil.copy(best_weights_src, target_weights_dst)
        print(
            f"\nAddestramento completato! Pesi salvati in: '{target_weights_dst}'"
        )
    else:
        print(
            f"\nWarning: Impossibile trovare il file dei pesi in {best_weights_src}"
        )


if __name__ == "__main__":
    train_model(epochs=20)