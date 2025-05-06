import os
import torch
import numpy as np
import pandas as pd
import argparse
import json
import matplotlib.pyplot as plt
from torchvision.transforms import Normalize, Compose, ToTensor
from drp.utils.backbone import generate_model
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score


class MinicubeDataset(Dataset):
    def __init__(self, csv_path, root_dir, cube_shape=(100, 100, 100), dtype=np.float32):
        """
        Args:
            csv_path (str): Path to the CSV file with filenames, offsets, and labels.
            root_dir (str): Directory with the .dat minicube files.
            cube_shape (tuple): Shape of the cubes (default: (100, 100, 100)).
            transform (callable, optional): Optional transform to apply on a sample.
            dtype (np.dtype): Data type of the cube (e.g., np.float32).
        """
        self.data_info = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.cube_shape = cube_shape
        self.transform = Compose([
            ToTensor(),
            Normalize(32928.34934854074, 6935.238201002077)
        ])
        self.dtype = dtype

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        filename = row["filename"]
        label = row["label"]

        # Chargement du minicube
        cube_path = os.path.join(self.root_dir, filename)
        cube = np.fromfile(cube_path, dtype=self.dtype)

        # Reshape to (num_cubes, 1, 100, 100, 100)
        cube = cube.reshape(self.cube_shape)
        cube = self.transform(cube)

        label = torch.tensor(label, dtype=torch.float32)

        return cube.unsqueeze(0), label


def evaluate_model_on_minicubes(root_dir, checkpoint_path, batch_size=4, model_depth=18, mean=0, std=1):
    csv_path = os.path.join(root_dir, "minicubes_info.csv")
    dataset = MinicubeDataset(csv_path, root_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    params = torch.load(checkpoint_path, map_location="cpu")
    new_model_params = {}
    for k in list(params["model"].keys()):
        new_key = k.replace("module.", "")
        new_model_params[new_key] = params["model"][k]
    params["model"] = new_model_params

    backbone = generate_model(model_depth=18, n_classes=1)
    backbone.load_state_dict(params["model"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(device)

    backbone.eval()

    results = []
    with torch.no_grad():
        for i, (x, labels) in enumerate(dataloader):
            x = x.to(device)
            pred = backbone(x).cpu()
            results.append({
                "predicted": pred.cpu().numpy(),
                "labels": labels.cpu().numpy(),
            })

    # R² score
    all_labels = np.concatenate([item["labels"] for item in results])
    all_preds = np.concatenate([item["predicted"] for item in results])
    r2 = r2_score(all_labels, all_preds)

    # Scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(all_labels, all_preds, alpha=0.7)
    plt.plot([all_labels.min(), all_labels.max()],
             [all_labels.min(), all_labels.max()],
             linestyle='--', color='red', label='Ideal')

    plt.xlabel("Ground Truth Permeability (Label)", fontsize=12)
    plt.ylabel("Predicted Permeability", fontsize=12)
    plt.title(f"Prediction vs Ground Truth (R² = {r2:.4f})", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig(os.path.join(root_dir, "scatter_plot.png"))
    print(f"Scatter Plot saved to: {root_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a 3D model on minicube data.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the JSON config file (e.g., config.json)"
    )
    args = parser.parse_args()

    # Load JSON config
    with open(args.config, "r") as f:
        config = json.load(f)

    evaluate_model_on_minicubes(
        root_dir=config["root_dir"],
        checkpoint_path=config["checkpoint_path"],
        batch_size=config["batch_size"],
        model_depth=config["model_depth"],
        mean=config["mean"],
        std=config["std"]
    )