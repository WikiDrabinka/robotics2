from pathlib import Path
import numpy as np
import cv2

def load_image_folder(csv_path: Path) -> list[tuple[np.array, float, float]]:
    folder_path = csv_path.parent / csv_path.stem
    images = []
    with open(csv_path,"r") as csv_file:
        for line in csv_file.readlines():
            name, forward, turn = line.split(",")
            name = "0" * (4 - len(name)) + name
            forward, turn = float(forward), float(turn)
            image_path = folder_path / (name + ".jpg")
            image = cv2.imread(image_path)
            images.append((image, forward, turn))
    return images

def load_all_images(dataset_path: Path):
    images = []
    for file in dataset_path.glob("*.csv"):
        images.extend(load_image_folder(file))
    return images

if __name__ == "__main__":
    dataset_path = Path("./dataset")
    print(len(load_all_images(dataset_path)))
