import os
from pathlib import Path
import cv2
from ultralytics import YOLO


def run_inference_on_video(
    video_path: str,
    model_path: str = "models/ant_detector_v2.pt",
    output_folder: str = "outputs/processed_videos",
    conf_threshold: float = 0.5,
):
    """Esegue l'inferenza del modello su un video in input e salva il risultato."""
    if not os.path.exists(video_path):
        print(f"Errore: Il video '{video_path}' non esiste.")
        return

    if not os.path.exists(model_path):
        print(f"Errore: Il file dei pesi '{model_path}' non è stato trovato.")
        return

    # 1. Carica il modello addestrato
    print(f"Caricamento modello da '{model_path}'...")
    model = YOLO(model_path)

    # 2. Prepara la cartella di destinazione
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_name = Path(video_path).stem
    output_path = out_dir / f"{video_name}_detected.mp4"

    print(f"Elaborazione video: '{video_path}'...")

    # 3. Esegue la predizione frame per frame (o usa direttamente la pipeline video di YOLO)
    # Usiamo stream=True se il video è lungo, oppure processiamo direttamente con save=True
    results = model.predict(
        source=video_path,
        conf=conf_threshold,
        save=True,
        project=str(out_dir),
        name="detection_run",
        exist_ok=True,
    )

    print(f"\nInferenza completata!")
    print(f"I risultati sono stati salvati nella cartella: {out_dir / 'detection_run'}")


if __name__ == "__main__":
    # Sostituisci con il percorso di un video di test presente in raw_videos
    test_video = "data/raw_videos/ant-sample05.mp4"
    run_inference_on_video(test_video, conf_threshold=0.60)