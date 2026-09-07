import cv2
import os
from pathlib import Path


def extract_frames(
    video_path: str, output_dir: str, frame_interval: int = 15
) -> int:
    """Estragga frame da un video salvandoli come immagini per il dataset.

    :param video_path: Percorso del file video di input
    :param output_dir: Cartella dove salvare i frame estratti
    :param frame_interval: Salva 1 frame ogni N frame (default 15)
    :return: Numero totale di frame estratti
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Errore: Impossibile aprire il video {video_path}")
        return 0

    frame_count = 0
    saved_count = 0

    print(f"Estrazione frame da {video_path.name}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Salva solo 1 frame ogni frame_interval
        if frame_count % frame_interval == 0:
            frame_filename = (
                output_dir / f"{video_path.stem}_frame_{saved_count:04d}.jpg"
            )
            cv2.imwrite(str(frame_filename), frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(
        f"Completato! Estratti {saved_count} frame e salvati in '{output_dir}'."
    )
    return saved_count


if __name__ == "__main__":
    # Test rapido
    video_input = "data/raw_videos/ant-sample04.mp4"
    output_folder = "data/raw_images"

    if os.path.exists(video_input):
        extract_frames(video_input, output_folder, frame_interval=15)
    else:
        print(f"Inserisci un video in '{video_input}' per eseguire lo script.")