# DATA_PREP_BDD100K.PY
# BDD100K data preprocessing pipeline for self-supervised frame ordering.
# Adapted from data_prep.py (UCF-101) with key differences:
#   - BDD100K videos are .mov (H.264, 720p, 30fps) vs UCF-101 .avi (320x240, 25fps)
#   - Flat directory structure (train/val/test) vs class-based folders
#   - No action class labels (self-supervised task only needs frame ordering)
#   - More aggressive frame decimation and downscaling before optical flow

##################################################
# 1. IMPORTS
##################################################
from azure.storage.blob import BlobServiceClient, ContainerClient
import torch
import numpy as np
import random
import cv2
import imageio.v3 as iio
from PIL import Image
from torch.utils.data import Dataset
import io
import gc
import traceback
import json
from dotenv import load_dotenv
import os

##################################################
# 2. CONSTANTS
##################################################
load_dotenv()

AZURE_STORAGE_URL = os.getenv('AZURE_STORAGE_URL')
AZURE_STORAGE_KEY = os.getenv('AZURE_STORAGE_KEY')
AZURE_CONTAINER_NAME = os.getenv('AZURE_CONTAINER_NAME')

with open('config.json', 'r') as f:
    config = json.load(f)['data_prep_bdd100k']

PREPROCESSEDDATA_FOLDERNAME = config['data']['preprocessed_folder']
BATCH_SIZE = config['data']['batch_size']
VIDEO_LIMIT = config['data']['video_limit']
SPLIT = config['data']['split']

# Preprocessing parameters tuned for 720p driving videos
PATCH_SIZE = config['preprocessing']['patch_size']
FRAME_DECIMATION = config['preprocessing']['frame_decimation']
OPTICAL_FLOW_DOWNSCALE = tuple(config['preprocessing']['optical_flow_downscale'])
MARGIN = config['preprocessing']['margin']
SPATIAL_JITTER = config['preprocessing']['spatial_jitter']
DOWNSCALE_BEFORE_PATCH = tuple(config['preprocessing']['downscale_before_patch'])

##################################################
# 3. CLASSES
##################################################

class BDD100KBlobSamples:
    """
    Handles loading BDD100K videos from Azure Blob Storage.
    BDD100K structure in Azure: bdd100k/videos/{train,val,test}/*.mov
    """
    def __init__(self, split="train"):
        self.split = split
        self.prefix = f"bdd100k/videos/{split}/"

    def list_videos(self, container_client):
        """List all .mov video blobs in the given split."""
        blob_list = container_client.list_blobs(name_starts_with=self.prefix)
        video_names = [
            blob.name for blob in blob_list
            if blob.name.endswith('.mov')
        ]
        print(f"Found {len(video_names)} videos in {self.prefix}")
        return video_names

    def load_videos_generator(self, blob_service_client, container_name, video_limit=None):
        """
        Generator that yields one video at a time from Azure Blob Storage.
        Each yielded item is a dict: {'path': blob_name, 'data': raw_bytes}
        """
        print(f"\nInitializing BDD100K video generator (split={self.split})...")
        container_client = blob_service_client.get_container_client(container_name)
        video_names = self.list_videos(container_client)

        if video_limit is not None:
            video_names = video_names[:video_limit]
            print(f"Limited to {video_limit} videos")

        for i, blob_name in enumerate(video_names, 1):
            print(f"\rLoading video {i}/{len(video_names)}: {os.path.basename(blob_name)}", end="")
            blob_client = blob_service_client.get_blob_client(
                container=container_name, blob=blob_name
            )
            video_data = blob_client.download_blob().readall()
            yield {'path': blob_name, 'data': video_data}

        print(f"\nFinished loading {len(video_names)} videos.")


class BDD100KPreparedDataset(Dataset):
    """
    Wraps raw BDD100K video bytes into a dataset for preprocessing.
    Unlike UCF-101's PreparedDataset, there are no action class labels here --
    the self-supervised frame ordering task generates its own labels.
    """
    def __init__(self, videos, batch_size=5):
        self.video_data = []
        self.video_names = []
        self.batch_size = batch_size

        for video in videos:
            video_name = os.path.basename(video['path']).replace('.mov', '')
            self.video_names.append(video_name)
            self.video_data.append(video)

        self.video_batches = self._create_batches()

    def _create_batches(self):
        batches = []
        batch = []
        for i, (name, video) in enumerate(zip(self.video_names, self.video_data)):
            batch.append((name, video))
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []
        if batch:
            batches.append(batch)
        return batches

    def __getitem__(self, index):
        return self.video_batches[index]

    def __len__(self):
        return len(self.video_batches)


class BDD100KPreprocessedData(Dataset):
    """
    Preprocesses BDD100K video frames for the frame ordering task.
    Adapted from PreprocessedTemporalFourData with BDD100K-specific handling:
      - .mov format with rotation metadata handling
      - 720p resolution: downscale before patch selection
      - Higher frame decimation (every 3rd frame at 30fps → ~10fps)
      - Larger margins for 720p content
    """
    def __init__(self, video_list, patch_size=PATCH_SIZE,
                 frame_decimation=FRAME_DECIMATION,
                 flow_downscale=OPTICAL_FLOW_DOWNSCALE,
                 margin=MARGIN, spatial_jitter_dist=SPATIAL_JITTER,
                 downscale_before_patch=DOWNSCALE_BEFORE_PATCH):
        """
        Args:
            video_list: list of (video_name, video_dict) tuples
            patch_size: size of the cropped patch (default 160x160)
            frame_decimation: keep every Nth frame (default 3 for 30fps → 10fps)
            flow_downscale: (W, H) to downscale frames before optical flow
            margin: pixel margin from edges for patch selection
            spatial_jitter_dist: max jitter distance in pixels
            downscale_before_patch: (W, H) to downscale frames before patch selection
        """
        self.video_list = video_list
        self.patch_size = patch_size
        self.frame_decimation = frame_decimation
        self.flow_downscale = flow_downscale
        self.margin = margin
        self.sjdis = spatial_jitter_dist
        self.downscale_before_patch = downscale_before_patch

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        video_name, video = self.video_list[index]

        # Read frames from .mov video
        frames = self._read_mov_frames(video['data'])

        # Frame decimation: reduce temporal redundancy
        frames = frames[::self.frame_decimation]

        if len(frames) < 4:
            raise ValueError(
                f"Video {video_name} has only {len(frames)} frames after decimation "
                f"(need >= 4). Original frame count may be too low."
            )

        # Downscale frames for processing (720p → smaller for optical flow & patch selection)
        frames_downscaled = np.array([
            cv2.resize(frame, self.downscale_before_patch)
            for frame in frames
        ])

        # Compute optical flow weights for motion-aware frame selection
        weights, flows = self._compute_optical_flow_weights(frames_downscaled)

        # Select 4 frames based on optical flow weights (motion-aware selection)
        indices = np.random.choice(len(frames_downscaled), size=4, replace=False, p=weights)
        selected_frames = frames_downscaled[indices]

        # Normalize indices to their relative order
        ranked_indices = indices.argsort().argsort()

        # Save ordered frames for visualization
        ordered_indices = np.sort(indices)
        ordered_frames = frames_downscaled[ordered_indices]

        # Generate frame ordering label (self-supervised)
        frame_order_label, frames_canonical_order = self._get_frame_order_label(ranked_indices)

        # Random horizontal mirroring (50% chance)
        if random.randint(0, 1) == 1:
            selected_frames = np.array([
                np.array(Image.fromarray(f.astype('uint8')).transpose(Image.FLIP_LEFT_RIGHT))
                for f in selected_frames
            ])
            ordered_frames = np.array([
                np.array(Image.fromarray(f.astype('uint8')).transpose(Image.FLIP_LEFT_RIGHT))
                for f in ordered_frames
            ])

        # Select the best motion patch
        best_patch = self._select_best_patch(selected_frames)
        if best_patch is None:
            # Fallback: center crop if no valid patch found
            h, w = selected_frames.shape[1], selected_frames.shape[2]
            best_patch = (
                max(0, (h - self.patch_size) // 2),
                max(0, (w - self.patch_size) // 2)
            )

        # Preprocess: spatial jittering + patch extraction
        preprocessed_frames, preprocessed_coords = self._preprocess_frames(
            selected_frames, best_patch
        )

        # Return the same 8-tuple format as UCF-101 pipeline for compatibility.
        # action_label is set to -1 (not applicable for BDD100K self-supervised task).
        action_label = torch.tensor(-1)
        return (
            preprocessed_frames,
            frame_order_label,
            action_label,
            video_name,
            frames_canonical_order,
            selected_frames,
            preprocessed_coords,
            ordered_frames
        )

    def _read_mov_frames(self, video_bytes):
        """
        Read frames from .mov video bytes.
        Handles BDD100K rotation metadata (many dashcam videos have rotate:270).
        """
        try:
            frames = iio.imread(video_bytes, index=None, format_hint=".mov")
        except Exception:
            # Fallback: try without format hint (let imageio auto-detect)
            frames = iio.imread(video_bytes, index=None)

        # BDD100K videos may need rotation correction.
        # If frames come out sideways (height > width significantly), rotate.
        if len(frames) > 0 and frames.shape[1] > frames.shape[2] * 1.5:
            frames = np.array([cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE) for f in frames])

        return frames

    def _compute_optical_flow_weights(self, frames):
        """Compute optical flow-based weights for motion-aware frame selection."""
        # Downscale further for optical flow computation
        downsampled = [cv2.resize(f, self.flow_downscale) for f in frames]
        gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f
                for f in downsampled]

        flows = [
            cv2.calcOpticalFlowFarneback(
                gray[i], gray[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            for i in range(len(gray) - 1)
        ]

        magnitudes = [np.sqrt(flow[..., 0]**2 + flow[..., 1]**2) for flow in flows]
        avg_magnitudes = [np.mean(mag) for mag in magnitudes]
        avg_magnitudes.append(0)  # Last frame has no forward flow

        total = np.sum(avg_magnitudes)
        if total == 0:
            # Uniform weights if no motion detected (static scene)
            weights = np.ones(len(frames)) / len(frames)
        else:
            weights = np.array(avg_magnitudes) / total

        return weights, flows

    def _get_frame_order_label(self, order_indices):
        """Generate canonical frame ordering label (same as UCF-101 pipeline)."""
        frame_order_to_label_dict = {
            (0, 1, 2, 3): 0,  (0, 2, 1, 3): 1,  (0, 3, 2, 1): 2,
            (0, 1, 3, 2): 3,  (0, 3, 1, 2): 4,  (0, 2, 3, 1): 5,
            (1, 0, 2, 3): 6,  (1, 0, 3, 2): 7,  (1, 2, 0, 3): 8,
            (1, 3, 0, 2): 9,  (2, 0, 1, 3): 10, (2, 1, 0, 3): 11
        }
        canonical = order_indices if order_indices[0] < order_indices[-1] else order_indices[::-1]
        label = frame_order_to_label_dict[tuple(canonical)]
        return torch.tensor(label), torch.tensor(canonical.copy())

    def _select_best_patch(self, selected_frames):
        """Find the 160x160 patch with maximum optical flow magnitude."""
        gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f
                for f in selected_frames]
        flows = [
            cv2.calcOpticalFlowFarneback(
                gray[i], gray[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            for i in range(len(gray) - 1)
        ]
        magnitudes = [np.sqrt(flow[..., 0]**2 + flow[..., 1]**2) for flow in flows]
        summed_flow = sum(magnitudes)

        best_patch = None
        best_motion_sum = -1

        h, w = summed_flow.shape
        for i in range(self.margin, h - self.patch_size - self.margin + 1):
            for j in range(self.margin, w - self.patch_size - self.margin + 1):
                motion_sum = summed_flow[i:i+self.patch_size, j:j+self.patch_size].sum()
                if motion_sum > best_motion_sum:
                    best_patch = (i, j)
                    best_motion_sum = motion_sum

        return best_patch

    def _preprocess_frames(self, selected_frames, best_patch):
        """Extract patches with spatial jittering."""
        startx, starty = best_patch
        preprocessed_frames = []
        preprocessed_coords = []

        for frame in selected_frames:
            shift_x = np.random.randint(-self.sjdis, self.sjdis)
            shift_y = np.random.randint(-self.sjdis, self.sjdis)

            newx = max(0, min(startx + shift_x, frame.shape[0] - self.patch_size))
            newy = max(0, min(starty + shift_y, frame.shape[1] - self.patch_size))

            patch = frame[newx:newx+self.patch_size, newy:newy+self.patch_size]
            preprocessed_frames.append(torch.from_numpy(patch.copy()))
            preprocessed_coords.append((newx, newy))

        return torch.stack(preprocessed_frames), preprocessed_coords


##################################################
# 4. HELPER FUNCTIONS
##################################################

def get_memory_usage():
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def log_memory(message):
    print(f"{message} - Memory usage: {get_memory_usage():.2f} MB")


def process_batch_fully_extracted(batch, batch_count, blob_service_client_instance):
    """
    Process a batch of raw BDD100K videos into pre-extracted 8-tuple samples.
    Saves to Azure Blob Storage as .pth files.
    """
    try:
        # Build list of (video_name, video_dict) for the preprocessor
        video_list = []
        for video in batch:
            video_name = os.path.basename(video['path']).replace('.mov', '')
            video_list.append((video_name, video))

        # Create the preprocessing dataset
        dataset = BDD100KPreprocessedData(video_list)

        # Extract all samples
        final_samples = []
        for idx in range(len(dataset)):
            try:
                sample = dataset[idx]
                final_samples.append(sample)
            except Exception as e:
                print(f"\n  [Warning] Error extracting video {idx} "
                      f"({video_list[idx][0]}): {str(e)}")
                continue

        if not final_samples:
            print(f"\n  [Warning] No valid samples in batch {batch_count}")
            return False

        print(f"\n  Extracted {len(final_samples)} samples from batch {batch_count}")

        # Serialize and upload to Azure
        buffer = io.BytesIO()
        torch.save(final_samples, buffer, pickle_protocol=5)
        buffer.seek(0)

        blob_client = blob_service_client_instance.get_blob_client(
            container=AZURE_CONTAINER_NAME,
            blob=f"{PREPROCESSEDDATA_FOLDERNAME}/bdd100k_preprocessed_fullyextracted_batch_{batch_count}.pth"
        )
        blob_client.upload_blob(buffer, overwrite=True)

        # Cleanup
        del dataset, final_samples, buffer
        gc.collect()
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"\n  [Error] Batch {batch_count}: {str(e)}")
        traceback.print_exc()
        return False


##################################################
# 5. MAIN EXECUTION
##################################################
if __name__ == "__main__":
    try:
        print("=" * 60)
        print("BDD100K Data Preprocessing Pipeline")
        print("=" * 60)
        log_memory("Initial memory usage")

        print(f"\nConfiguration:")
        print(f"  Split: {SPLIT}")
        print(f"  Batch size: {BATCH_SIZE}")
        print(f"  Video limit: {VIDEO_LIMIT}")
        print(f"  Frame decimation: every {FRAME_DECIMATION}th frame")
        print(f"  Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
        print(f"  Downscale before patch: {DOWNSCALE_BEFORE_PATCH}")
        print(f"  Optical flow downscale: {OPTICAL_FLOW_DOWNSCALE}")
        print(f"  Margin: {MARGIN}")
        print(f"  Spatial jitter: +/-{SPATIAL_JITTER}")
        print(f"  Output folder: {PREPROCESSEDDATA_FOLDERNAME}")

        # Initialize Azure connection
        print("\nConnecting to Azure Blob Storage...")
        blob_service_client = BlobServiceClient(
            account_url=AZURE_STORAGE_URL, credential=AZURE_STORAGE_KEY
        )

        # Initialize BDD100K video loader
        bdd_samples = BDD100KBlobSamples(split=SPLIT)

        video_generator = bdd_samples.load_videos_generator(
            blob_service_client, AZURE_CONTAINER_NAME, video_limit=VIDEO_LIMIT
        )

        # Process videos in batches
        print("\nStarting batch preprocessing...")
        batch = []
        batch_count = 0
        video_count = 0
        failed_videos = []

        for video in video_generator:
            try:
                batch.append(video)
                video_count += 1

                if len(batch) == BATCH_SIZE:
                    batch_count += 1
                    print(f"\n\nProcessing batch {batch_count} ({BATCH_SIZE} videos)...")
                    success = process_batch_fully_extracted(
                        batch, batch_count, blob_service_client
                    )
                    if not success:
                        print(f"  Failed batch {batch_count}")
                        failed_videos.extend(batch)
                    batch = []
                    log_memory(f"  After batch {batch_count}")

            except Exception as e:
                print(f"\nError processing video {video_count}: {str(e)}")
                failed_videos.append(video)
                continue

        # Process remaining videos
        if batch:
            try:
                batch_count += 1
                print(f"\n\nProcessing final batch {batch_count} ({len(batch)} videos)...")
                success = process_batch_fully_extracted(
                    batch, batch_count, blob_service_client
                )
                if not success:
                    failed_videos.extend(batch)
            except Exception as e:
                print(f"\nError processing final batch: {str(e)}")
                failed_videos.extend(batch)

        # Report
        print("\n" + "=" * 60)
        print("BDD100K Preprocessing Complete")
        print("=" * 60)
        print(f"Total videos processed: {video_count}")
        print(f"Total batches created: {batch_count}")
        print(f"Failed videos: {len(failed_videos)}")
        log_memory("Final memory usage")

    except Exception as e:
        print(f"\nCritical error: {str(e)}")
        traceback.print_exc()
    finally:
        print("\nProcess finished.")
