"""
Diagnostic script to check if BDD100K preprocessed data in Azure Blob Storage
is compatible with bdd100k_main.py's expected format.

Run: python check_bdd100k_data.py

Checks:
  1. Can we find .pth files in the configured folder?
  2. Do filenames contain "fullyextracted" (required by the training script)?
  3. Is each .pth file a list of tuples?
  4. Are tensor shapes correct (4, 160, 160, 3) for the model input?
  5. Are frame order labels valid integers 0-11?
"""
import os
import json
import torch
import io
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

load_dotenv()

AZURE_STORAGE_URL = os.getenv('AZURE_STORAGE_URL')
AZURE_STORAGE_KEY = os.getenv('AZURE_STORAGE_KEY')
AZURE_CONTAINER_NAME = os.getenv('AZURE_CONTAINER_NAME')

with open('config.json', 'r') as f:
    config = json.load(f)['main_bdd100k']
    FOLDER = config['paths']['preprocessed_folder']


def main():
    print("=" * 60)
    print("BDD100K Preprocessed Data Compatibility Check")
    print("=" * 60)

    # Connect to Azure
    print(f"\nConnecting to Azure Blob Storage...")
    print(f"  Storage URL: {AZURE_STORAGE_URL}")
    print(f"  Container: {AZURE_CONTAINER_NAME}")
    print(f"  Folder: {FOLDER}/")

    blob_service_client = BlobServiceClient(
        account_url=AZURE_STORAGE_URL, credential=AZURE_STORAGE_KEY
    )
    container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)

    # Step 1: List all blobs in folder
    print(f"\n--- Step 1: Listing blobs in '{FOLDER}/' ---")
    blob_list = list(container_client.list_blobs(name_starts_with=FOLDER + '/'))
    all_blobs = [b.name for b in blob_list]
    pth_blobs = [b for b in all_blobs if b.endswith('.pth')]

    print(f"  Total blobs found: {len(all_blobs)}")
    print(f"  .pth files found: {len(pth_blobs)}")

    if not pth_blobs:
        print("\n  [FAIL] No .pth files found! The training script won't find any data.")
        print("  Files in folder:")
        for b in all_blobs[:20]:
            print(f"    {b}")
        return

    # Show first few filenames
    print(f"\n  First 5 .pth files:")
    for b in pth_blobs[:5]:
        print(f"    {b}")

    # Step 2: Check "fullyextracted" in filenames
    print(f"\n--- Step 2: Checking filename filter ('fullyextracted' keyword) ---")
    fullyextracted_blobs = [b for b in pth_blobs if "fullyextracted" in b]
    print(f"  Files with 'fullyextracted': {len(fullyextracted_blobs)} / {len(pth_blobs)}")

    if not fullyextracted_blobs:
        print("\n  [FAIL] No files contain 'fullyextracted' in the name!")
        print("  bdd100k_main.py filters for: blob.name.endswith('.pth') and 'fullyextracted' in blob.name")
        print("  Your files are named:")
        for b in pth_blobs[:10]:
            print(f"    {os.path.basename(b)}")
        print("\n  OPTIONS:")
        print("  a) Rename files to include 'fullyextracted' in the name")
        print("  b) Modify bdd100k_main.py to remove the 'fullyextracted' filter")
        # Continue checking data format with whatever .pth files exist
    else:
        print("  [OK] Filename filter will work.")

    # Step 3: Download and inspect one .pth file
    test_blob = pth_blobs[0]
    print(f"\n--- Step 3: Inspecting data format of '{os.path.basename(test_blob)}' ---")

    blob_client = blob_service_client.get_blob_client(
        container=AZURE_CONTAINER_NAME, blob=test_blob
    )
    downloaded = blob_client.download_blob().readall()
    print(f"  File size: {len(downloaded) / 1024:.1f} KB")

    data = torch.load(io.BytesIO(downloaded), weights_only=False)

    # Check top-level structure
    print(f"\n  Top-level type: {type(data).__name__}")

    if isinstance(data, list):
        print(f"  [OK] Is a list (expected)")
        print(f"  Number of samples: {len(data)}")

        if len(data) == 0:
            print("  [FAIL] List is empty!")
            return

        # Inspect first sample
        sample = data[0]
        print(f"\n  First sample type: {type(sample).__name__}")

        if isinstance(sample, (tuple, list)):
            print(f"  [OK] Sample is a tuple/list")
            print(f"  Number of elements: {len(sample)}")
            print(f"  Expected: 8 (preprocessed_frames, frame_order_label, action_label, "
                  f"video_name, canonical_order, selected_frames, coords, ordered_frames)")

            if len(sample) >= 2:
                inputs = sample[0]
                label = sample[1]

                print(f"\n  Element [0] (preprocessed_frames):")
                print(f"    Type: {type(inputs).__name__}")
                if hasattr(inputs, 'shape'):
                    print(f"    Shape: {inputs.shape}")
                    print(f"    Dtype: {inputs.dtype}")
                    expected_shape = (4, 160, 160, 3)
                    if tuple(inputs.shape) == expected_shape:
                        print(f"    [OK] Shape matches expected {expected_shape}")
                    else:
                        print(f"    [FAIL] Shape mismatch! Expected {expected_shape}, got {tuple(inputs.shape)}")
                        print(f"    The training loop does: inputs.permute(0,1,4,2,3) then view(-1, 12, 160, 160)")
                        if len(inputs.shape) == 4 and inputs.shape[0] == 4:
                            h, w = inputs.shape[1], inputs.shape[2]
                            c = inputs.shape[3] if len(inputs.shape) > 3 else 'N/A'
                            print(f"    Your data: 4 frames of {h}x{w} with {c} channels")
                            if h != 160 or w != 160:
                                print(f"    [FAIL] Patch size must be 160x160 for the model (conv layers expect this)")
                else:
                    print(f"    Value: {inputs}")
                    print(f"    [FAIL] Expected a tensor with shape, got {type(inputs).__name__}")

                print(f"\n  Element [1] (frame_order_label):")
                print(f"    Type: {type(label).__name__}")
                if hasattr(label, 'item'):
                    val = label.item()
                    print(f"    Value: {val}")
                    if 0 <= val <= 11:
                        print(f"    [OK] Valid frame order label (0-11)")
                    else:
                        print(f"    [FAIL] Expected 0-11, got {val}")
                elif isinstance(label, int):
                    print(f"    Value: {label}")
                    if 0 <= label <= 11:
                        print(f"    [WARNING] Label is int, not tensor. DataLoader may handle this, but tensor is expected.")
                    else:
                        print(f"    [FAIL] Expected 0-11, got {label}")
                else:
                    print(f"    Value: {label}")
                    print(f"    [FAIL] Expected tensor scalar or int, got {type(label).__name__}")

                # Check remaining elements
                if len(sample) >= 3:
                    print(f"\n  Element [2] (action_label): type={type(sample[2]).__name__}, value={sample[2] if not hasattr(sample[2], 'shape') else f'tensor({sample[2].item()})'}")
                if len(sample) >= 4:
                    print(f"  Element [3] (video_name): type={type(sample[3]).__name__}, value={sample[3][:50] if isinstance(sample[3], str) else sample[3]}")

            else:
                print(f"  [FAIL] Sample has fewer than 2 elements. Training loop needs at least (inputs, label, ...).")

        elif isinstance(sample, dict):
            print(f"  [FAIL] Sample is a dict, but training expects a tuple/list.")
            print(f"  Keys: {list(sample.keys())[:10]}")
            print(f"  The DataLoader unpacks samples as: (inputs, frame_order_labels, *rest)")
            print(f"  Dicts can't be unpacked this way.")

        else:
            print(f"  [FAIL] Expected tuple or list, got {type(sample).__name__}")

    elif isinstance(data, dict):
        print(f"  [FAIL] File contains a dict, not a list.")
        print(f"  Keys: {list(data.keys())[:10]}")
        print(f"  bdd100k_main.py expects: list of 8-tuples")

    elif isinstance(data, torch.Tensor):
        print(f"  [FAIL] File contains a single tensor of shape {data.shape}")
        print(f"  bdd100k_main.py expects: list of 8-tuples")

    else:
        print(f"  [FAIL] Unexpected data type: {type(data).__name__}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Folder path match: {'OK' if FOLDER in test_blob else 'CHECK'}")
    print(f"Files found: {len(pth_blobs)}")
    print(f"'fullyextracted' filter: {'OK' if fullyextracted_blobs else 'FAIL - files will be filtered out!'}")
    print(f"Run with: python check_bdd100k_data.py")


if __name__ == '__main__':
    main()
