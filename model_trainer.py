import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# Import the data loading function from your file
from DataLoader import load_all_images


# Model architecture matching your autonomous.py
class TinyJetbotNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.BatchNorm2d(24), nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.BatchNorm2d(36), nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.BatchNorm2d(48), nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 5 * 5, 100), nn.BatchNorm1d(100), nn.ReLU(),
            nn.Linear(100, 50), nn.BatchNorm1d(50), nn.ReLU(),
            nn.Linear(50, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return torch.tanh(x)


# PyTorch Dataset for handling the loaded images
class JetBotDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_bgr, forward, turn = self.data[idx]

        # Preprocessing matching autonomous.py
        # 1. Convert BGR to RGB (DataLoader loads with cv2.imread which is BGR)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2. Resize to 224x224
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

        # 3. Normalize ([0, 1] then apply ImageNet mean and std)
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std

        # 4. Convert HWC (Height, Width, Channels) to CHW (Channels, Height, Width)
        img = np.transpose(img, (2, 0, 1))

        # autonomous.py maps model output to: forward=out[1], left=out[0]
        # Therefore, target should be [turn, forward]
        target = np.array([turn, forward], dtype=np.float32)

        return torch.from_numpy(img), torch.from_numpy(target)


def train_and_export():
    dataset_path = Path("./dataset")
    print(f"Loading data from {dataset_path.absolute()}...")
    raw_data = load_all_images(dataset_path)

    if len(raw_data) == 0:
        print("No training data found. Make sure your dataset path has the expected CSVs and images.")
        return

    print(f"Loaded {len(raw_data)} images. Setting up model...")

    # Hyperparameters & Device
    batch_size = 32
    epochs = 20
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Prepare data loaders
    dataset = JetBotDataset(raw_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize model, loss (MSE for regression), optimizer
    model = TinyJetbotNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop with tqdm
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Wrap the dataloader with tqdm for a progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")

        for images, targets in progress_bar:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Update the progress bar text with the current average loss
            current_loss = running_loss / (progress_bar.n + 1)
            progress_bar.set_postfix(loss=f"{current_loss:.4f}")

    # 1. Save standard PyTorch weights
    pth_path = "best_jetbot_model.pth"
    torch.save(model.state_dict(), pth_path)
    print(f"Saved PyTorch weights to {pth_path}")

    # 2. Export to ONNX
    onnx_path = "model_final_final_02_final_please.onnx"
    model.eval()  # Set model to inference mode before export

    # We create a dummy input of the correct size: (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Successfully exported model to ONNX: {onnx_path}")


if __name__ == '__main__':
    train_and_export()