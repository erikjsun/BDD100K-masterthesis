# BDD100K_MAIN.PY
# Training pipeline for COPN on BDD100K preprocessed data.
# Uses the same model architecture and training loop as UCF-101 (main.py),
# but reads from BDD100K preprocessed .pth files.

##################################################
# 1. IMPORTS
##################################################
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt
import gc
import random
from sklearn.model_selection import train_test_split
from azure.storage.blob import BlobServiceClient
import io
from dotenv import load_dotenv

##################################################
# 2. CUSTOM IMPORTS
##################################################
from model import CustomOPN

##################################################
# 3. GLOBAL CONFIG
##################################################
load_dotenv()

AZURE_STORAGE_URL = os.getenv('AZURE_STORAGE_URL')
AZURE_STORAGE_KEY = os.getenv('AZURE_STORAGE_KEY')
AZURE_CONTAINER_NAME = os.getenv('AZURE_CONTAINER_NAME')

with open('config.json', 'r') as f:
    config = json.load(f)['main_bdd100k']
    PREPROCESSEDDATA_FOLDERNAME = config['paths']['preprocessed_folder']

epoch_amount = config['training']['epochs']
chunk_size = config['training']['chunk_size']
training_batch_size = config['training']['batch_size']
num_workers = config['training']['num_workers']

##################################################
# 4. DATASET CLASS FOR FULLY EXTRACTED .PTH FILES
##################################################
class FullyExtractedBlobDataset(Dataset):
    """
    Loads pre-extracted .pth files from Azure Blob Storage.
    Identical to main.py's version -- the .pth format is the same
    for both UCF-101 and BDD100K preprocessed data.
    """
    MAX_CACHE_SIZE_GB = 18

    def __init__(self, blob_service_client, container_name, pth_blob_names, cache_dir="local_cache_bdd100k"):
        self.blob_service_client = blob_service_client
        self.container_name = container_name
        self.pth_blob_names = pth_blob_names
        self.cache_dir = cache_dir
        self.samples = []

        os.makedirs(self.cache_dir, exist_ok=True)
        self._load_pth_files()

    def _get_cache_size_gb(self):
        total_size = 0
        if os.path.exists(self.cache_dir):
            for dirpath, dirnames, filenames in os.walk(self.cache_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        return total_size / (1024**3)

    def _load_pth_files(self):
        for i, blob_name in enumerate(self.pth_blob_names, start=1):
            cache_filename = blob_name.replace('/', '_')
            cache_path = os.path.join(self.cache_dir, cache_filename)

            if os.path.exists(cache_path):
                print(f"    => Loading from cache {i}/{len(self.pth_blob_names)}: {blob_name}")
                load_start = time.time()
                data_in_this_file = torch.load(cache_path, weights_only=False)
                print(f"       Cache load took {time.time() - load_start:.2f} seconds.")
            else:
                print(f"    => Downloading file {i}/{len(self.pth_blob_names)}: {blob_name}")
                dl_start = time.time()
                blob_client = self.blob_service_client.get_blob_client(
                    container=self.container_name, blob=blob_name
                )
                downloaded_blob = blob_client.download_blob().readall()
                print(f"       Download took {time.time() - dl_start:.2f} seconds. "
                      f"Size: {len(downloaded_blob)} bytes.")

                buffer = io.BytesIO(downloaded_blob)
                data_in_this_file = torch.load(buffer, weights_only=False)

                current_cache_size = self._get_cache_size_gb()
                file_size_gb = len(downloaded_blob) / (1024**3)

                if current_cache_size + file_size_gb <= self.MAX_CACHE_SIZE_GB:
                    print(f"       Saving to cache ({current_cache_size:.1f}GB/{self.MAX_CACHE_SIZE_GB}GB used)")
                    with open(cache_path, 'wb') as f:
                        f.write(downloaded_blob)
                else:
                    print(f"       Cache full ({current_cache_size:.1f}GB/{self.MAX_CACHE_SIZE_GB}GB), skipping")

            self.samples.extend(data_in_this_file)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

##################################################
# 5. HELPER FUNCTIONS
##################################################
def chunkify(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]

def train_one_chunk(model, criterion, optimizer, train_loader, current_chunk, total_chunks):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for batch_idx, (inputs, frame_order_labels, *rest) in enumerate(train_loader):
        inputs = inputs.float().permute(0, 1, 4, 2, 3).contiguous()
        inputs = inputs.view(-1, inputs.shape[1] * inputs.shape[2], inputs.shape[3], inputs.shape[4])

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, frame_order_labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += (preds == frame_order_labels).sum().item()
        total_samples += inputs.size(0)

    chunk_loss = running_loss / total_samples
    return chunk_loss, running_corrects, total_samples

def validate_in_chunks(model, blob_service_client, val_blob_names, batch_size, chunk_size):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for blob_chunk_idx, blob_chunk in enumerate(chunkify(val_blob_names, chunk_size), start=1):
        print(f"Validating on blob_chunk {blob_chunk_idx} with {len(blob_chunk)} file(s)...")

        blob_chunk_start = time.time()
        val_dataset = FullyExtractedBlobDataset(
            blob_service_client, AZURE_CONTAINER_NAME, blob_chunk
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        print(f"    Blob chunk {blob_chunk_idx} loaded in {time.time() - blob_chunk_start:.2f} sec")

        with torch.no_grad():
            for inputs, frame_order_labels, *rest in val_loader:
                inputs = inputs.float().permute(0, 1, 4, 2, 3).contiguous()
                inputs = inputs.view(-1, inputs.shape[1] * inputs.shape[2], inputs.shape[3], inputs.shape[4])

                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, frame_order_labels)

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == frame_order_labels).sum().item()
                total_samples += inputs.size(0)

        del val_dataset, val_loader
        gc.collect()
        torch.cuda.empty_cache()

    val_loss = running_loss / total_samples
    val_acc = running_corrects / total_samples
    return val_loss, val_acc

def plot_loss_and_accuracy(train_loss_history, train_accuracy_history,
                           val_loss_history, val_accuracy_history, plot_save_path):
    epochs = range(1, len(train_loss_history) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    axes[0].plot(epochs, train_loss_history, 'b-', label='Train Loss')
    axes[0].plot(epochs, val_loss_history, 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('BDD100K COPN Training - Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, train_accuracy_history, 'b-', label='Train Accuracy')
    axes[1].plot(epochs, val_accuracy_history, 'r-', label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('BDD100K COPN Training - Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(plot_save_path)
    print(f"Plot saved to {plot_save_path}")

##################################################
# 6. TRAINING PIPELINE
##################################################
def train_model(model_save_path, blob_service_client,
                train_blob_names, val_blob_names,
                epochs=30, chunk_size=2, batch_size=32):

    model = CustomOPN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.999), weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    # Save directory
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    snapshot_dir = os.path.join(os.path.dirname(model_save_path), 'bdd100k_modelsnapshots')
    os.makedirs(snapshot_dir, exist_ok=True)

    print("\nStarting BDD100K Training Loop...")
    print(f"  Epochs: {epochs}")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Train files: {len(train_blob_names)}")
    print(f"  Val files: {len(val_blob_names)}")

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")

        train_files_shuffled = train_blob_names[:]
        random.shuffle(train_files_shuffled)
        total_chunks = (len(train_files_shuffled) + chunk_size - 1) // chunk_size

        epoch_train_loss = 0.0
        epoch_train_corrects = 0
        epoch_train_samples = 0

        for blob_chunk_idx, blob_chunk in enumerate(chunkify(train_files_shuffled, chunk_size), start=1):
            print(f"\nProcessing train chunk {blob_chunk_idx}/{total_chunks}")
            print(f"  Loading {len(blob_chunk)} file(s)...")

            load_start = time.time()
            train_dataset = FullyExtractedBlobDataset(
                blob_service_client, AZURE_CONTAINER_NAME, blob_chunk
            )
            print(f"  Chunk loaded in {time.time() - load_start:.2f} seconds")

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
            )

            train_start = time.time()
            chunk_loss, chunk_corrects, chunk_samples = train_one_chunk(
                model, criterion, optimizer, train_loader,
                current_chunk=blob_chunk_idx, total_chunks=total_chunks
            )
            print(f"  Chunk {blob_chunk_idx}/{total_chunks} trained in {time.time() - train_start:.2f} sec")

            epoch_train_loss += chunk_loss * chunk_samples
            epoch_train_corrects += chunk_corrects
            epoch_train_samples += chunk_samples

            del train_dataset, train_loader
            gc.collect()
            torch.cuda.empty_cache()

        epoch_train_loss = epoch_train_loss / epoch_train_samples
        epoch_train_acc = epoch_train_corrects / epoch_train_samples

        val_loss, val_acc = validate_in_chunks(
            model, blob_service_client, val_blob_names, batch_size, chunk_size
        )

        train_loss_history.append(epoch_train_loss)
        train_acc_history.append(epoch_train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        print(f"Epoch {epoch+1} => Train Loss: {epoch_train_loss:.4f}, "
              f"Train Acc: {epoch_train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        scheduler.step()

        # Save snapshot every 10 epochs
        if (epoch + 1) % 10 == 0:
            snapshot_path = os.path.join(snapshot_dir, f'bdd100k_model_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), snapshot_path)
            print(f"  Snapshot saved: {snapshot_path}")

    # Save final model
    torch.save(model.state_dict(), model_save_path)
    return model, train_loss_history, train_acc_history, val_loss_history, val_acc_history

##################################################
# 7. MAIN EXECUTION
##################################################
def main():
    model_save_path = config['paths']['model_save']
    plot_save_path = config['paths']['plot_save']

    blob_service_client = BlobServiceClient(
        account_url=AZURE_STORAGE_URL, credential=AZURE_STORAGE_KEY
    )
    container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)

    # List BDD100K preprocessed .pth files
    blob_list = container_client.list_blobs(
        name_starts_with=PREPROCESSEDDATA_FOLDERNAME + '/'
    )
    all_blob_names = [
        blob.name for blob in blob_list
        if blob.name.endswith('.pth') and "fullyextracted" in blob.name
    ]

    if not all_blob_names:
        print("No BDD100K preprocessed .pth files found in blob storage.")
        print(f"Expected folder: {PREPROCESSEDDATA_FOLDERNAME}/")
        print("Run data_prep_bdd100k.py first to preprocess the videos.")
        return

    print(f"Found {len(all_blob_names)} BDD100K preprocessed .pth files.")

    # Train/Val split at file level
    train_blob_names, val_blob_names = train_test_split(
        all_blob_names, test_size=0.2, random_state=42
    )
    print(f"{len(train_blob_names)} train files, {len(val_blob_names)} val files.")

    # Train
    model, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = train_model(
        model_save_path=model_save_path,
        blob_service_client=blob_service_client,
        train_blob_names=train_blob_names,
        val_blob_names=val_blob_names,
        epochs=epoch_amount,
        chunk_size=chunk_size,
        batch_size=training_batch_size
    )

    # Plot
    plot_loss_and_accuracy(
        train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist, plot_save_path
    )

if __name__ == '__main__':
    main()
