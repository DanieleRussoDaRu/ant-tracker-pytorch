from pathlib import Path
import shutil
from ultralytics import YOLO


def train_model(
    data_yaml: str = "dataset.yaml",
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 8,
    weights: str = "yolo11n.pt",
):
    print(
        f"--- Inizio Addestramento Modello V2 su Nuovo Dataset ({epochs} epoche) ---"
    )

    model = YOLO(weights)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=0,
        name="ant_detector_v2",
        project="outputs/runs",
        optimizer="auto",  # Ripristina l'ottimizzatore di default (SGD/Auto)
        lr0=0.01,  # Learning rate standard per YOLO
        amp=False,  # <-- FONDAMENTALE: Disattiva FP16 per evitare Loss NaN
        plots=True,
    )

    save_dir = Path(results.save_dir)
    best_weights_src = save_dir / "weights" / "best.pt"
    target_weights_dst = Path("models/ant_detector_v2_best.pt")

    if best_weights_src.exists():
        target_weights_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_weights_src, target_weights_dst)
        print(
            f"\nAddestramento V2 completato! Nuovi pesi salvati in: '{target_weights_dst}'"
        )
    else:
        print(
            f"\nWarning: Impossibile trovare il file dei pesi in {best_weights_src}"
        )


if __name__ == "__main__":
    train_model(epochs=50)