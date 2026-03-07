import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))
import argparse
import numpy as np
import wandb

from utils.data_loader import load_data


CLASS_NAMES = [
    "0","1","2","3","4",
    "5","6","7","8","9"
]


def collect_samples(X, y, samples_per_class=5):

    labels = np.argmax(y, axis=1)

    samples = {i: [] for i in range(10)}

    for img, label in zip(X, labels):

        if len(samples[label]) < samples_per_class:
            samples[label].append(img)

        if all(len(v) == samples_per_class for v in samples.values()):
            break

    return samples


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--wandb_project", default="da6401")
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, name="task1-data-exploration")

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.dataset)

    X = np.vstack([X_train, X_val, X_test])
    y = np.vstack([y_train, y_val, y_test])

    samples = collect_samples(X, y)

    table = wandb.Table(columns=["class","img1","img2","img3","img4","img5"])

    for c, imgs in samples.items():

        row = [CLASS_NAMES[c]]

        for img in imgs:
            row.append(wandb.Image(img.reshape(28,28)))

        table.add_data(*row)

    wandb.log({"dataset_samples": table})

    wandb.finish()


if __name__ == "__main__":
    main()