from pathlib import Path
import shutil
from ultralytics import YOLO


def train_model(
    data_yaml: str = "dataset.yaml",
    epochs: int = 50,  # Aumentato a 50 epoche per sfruttare il nuovo dataset
    imgsz: int = 640,
    batch_size: int = 8,  # Se la tua CPU/GPU lo supporta, 8 velocizza il calcolo
    weights: str = "yolo11n.pt",  # Riparti dai pesi base per la massima pulizia
):
    """Esegue il fine-tuning del modello YOLO per la detection delle formiche."""
    print(
        f"--- Inizio Addestramento Modello su Nuovo Dataset ({epochs} epoche) ---"
    )

    model = YOLO(weights)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        name="ant_detector_v2",
        project="outputs/runs",
        optimizer="AdamW",
        lr0=0.001,
        plots=True,
    )

    save_dir = Path(results.save_dir)
    best_weights_src = save_dir / "weights" / "best.pt"
    target_weights_dst = Path("models/ant_detector_best.pt")

    if best_weights_src.exists():
        target_weights_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_weights_src, target_weights_dst)
        print(
            f"\nAddestramento completato! Nuovi pesi salvati in: '{target_weights_dst}'"
        )
    else:
        print(
            f"\nWarning: Impossibile trovare il file dei pesi in {best_weights_src}"
        )


if __name__ == "__main__":
    train_model(epochs=50)