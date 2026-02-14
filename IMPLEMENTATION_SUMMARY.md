# COPN Implementation Summary: Recreating Order Prediction Networks

**For Master's Thesis Results Section**
*Comprehensive documentation of implementation decisions and technical approach*

---

## Executive Summary

This document summarizes the implementation of a Custom Order Prediction Network (COPN) based on Lee et al.'s "Unsupervised Representation Learning by Sorting Sequences" (2017). The implementation involved:

1. **Framework Modernization**: Converting the original Caffe implementation to PyTorch, reducing ~1,500 lines of configuration code to 78 lines of model code
2. **Data Pipeline Engineering**: Building a scalable preprocessing pipeline with Azure Blob Storage integration
3. **Memory Management**: Solving RAM constraints through batched processing and chunked training
4. **Infrastructure**: Establishing a reproducible development environment with Docker and cloud storage

The UCF-101 pipeline serves as a validated baseline before applying the methodology to the BDD100K autonomous driving dataset.

---

## 1. Framework Conversion: Caffe to PyTorch

### 1.1 Motivation

The original OPN implementation by Lee et al. used Caffe, a deep learning framework developed by Berkeley AI Research (BAIR). While Caffe was state-of-the-art in 2017, it has since been superseded by more flexible frameworks. The decision to convert to PyTorch was motivated by:

- **Maintainability**: Caffe's layer-by-layer `.prototxt` configuration files are verbose and difficult to debug
- **Ecosystem**: PyTorch offers superior tooling, documentation, and community support
- **Flexibility**: PyTorch's dynamic computation graphs simplify experimentation
- **Modern Best Practices**: Access to modern optimizers, schedulers, and data loading utilities

### 1.2 Code Reduction

| Component | Original Caffe | New PyTorch | Reduction |
|-----------|---------------|-------------|-----------|
| Model Architecture | 1,311 lines (`train_opn.prototxt`) | 78 lines (`model.py`) | **94%** |
| Data Layers | 220 lines (`UCF_datalayers.py`) | Integrated into `data_prep.py` | — |
| Solver Config | 65+ lines (`.prototxt` files) | 15 lines in `main.py` | **77%** |
| **Total Core Code** | ~1,600 lines | ~1,200 lines | **25%** |

The dramatic reduction in model code (1,311 → 78 lines) demonstrates PyTorch's expressiveness. The Caffe `.prototxt` format requires explicit specification of every layer connection, parameter initialization, and training/test phase behavior, while PyTorch's `nn.Module` abstraction handles these implicitly.

### 1.3 Architecture Preservation

The COPN architecture faithfully reproduces the OPN's three-stage design:

**Stage 1: Feature Extraction (AlexNet-derived)**
```
Conv1 → ReLU → MaxPool → LRN →
Conv2 → ReLU → MaxPool → LRN →
Conv3 → BatchNorm → ReLU →
Conv4 → BatchNorm → ReLU →
Conv5 → BatchNorm → ReLU → MaxPool
```

**Stage 2: Pairwise Feature Extraction**
- FC6 layer outputs 1024 features, split into 4 parts (256 each)
- 6 pairwise concatenations: (1,2), (2,3), (3,4), (1,3), (2,4), (1,4)
- Each pair processed by shared-weight FC7 layers (512 units)

**Stage 3: Order Prediction**
- FC8 layer: 3072 inputs (6×512) → 12 outputs
- 12 classes represent canonical frame orderings (24 permutations reduced by forward/backward equivalence)

### 1.4 Architectural Deviation: Batch Normalization

The original OPN used batch normalization selectively (conv3-5, fc6-7). The COPN implementation maintains this pattern but uses PyTorch's integrated `BatchNorm2d` and `BatchNorm1d` layers, which handle training/test phase switching automatically (vs. Caffe's explicit `use_global_stats` flag).

---

## 2. Data Preprocessing Pipeline

### 2.1 Overview

The preprocessing pipeline transforms raw UCF-101 videos into training-ready 4-frame tuples. Each tuple consists of:

1. **Preprocessed frames**: 4 × 160×160 pixel patches
2. **Frame order label**: Integer 0-11 (canonical ordering class)
3. **Action label**: Integer 0-100 (UCF-101 action class)
4. **Metadata**: Video name, coordinates, original frames (for visualization)

### 2.2 Motion-Aware Frame Selection

**Problem**: Randomly selecting frames risks choosing temporally adjacent frames with minimal visual difference, creating ambiguous ordering tasks.

**Solution**: Optical flow-weighted sampling

```python
# Farneback optical flow computation
flows = cv2.calcOpticalFlowFarneback(frame_i, frame_i+1, ...)
magnitudes = sqrt(flow_x² + flow_y²)
weights = avg_magnitudes / sum(avg_magnitudes)

# Weighted random selection of 4 frames
indices = np.random.choice(frames, size=4, p=weights)
```

**Implementation Decision**: The original paper's motion-aware selection details were sparse. This implementation uses the Farneback dense optical flow method, validated by Lyasheva et al. for frame interpolation tasks. Frames are downsampled to 160×80 before flow computation to reduce O(N²) complexity, with no observable quality degradation.

### 2.3 Motion-Aware Patch Selection

**Problem**: Full-frame processing is computationally expensive and may include static background regions with no temporal information.

**Solution**: Sliding window search for maximum-motion 160×160 patch

```python
# Sum optical flow magnitudes across frame transitions
summed_flow = sum(magnitudes_between_selected_frames)

# Exhaustive search with margin constraints
for i in range(margin, H - patch_size - margin):
    for j in range(margin, W - patch_size - margin):
        motion_sum = summed_flow[i:i+patch_size, j:j+patch_size].sum()
        if motion_sum > best_motion_sum:
            best_patch = (i, j)
```

**Implementation Decision**: A 30-pixel margin from frame edges prevents patch selection in areas that may be cropped or contain artifacts. The 160×160 patch size matches the original paper and balances computational efficiency with sufficient visual context.

### 2.4 Spatial Jittering

**Problem**: Without variation in patch position, the network might memorize absolute spatial locations rather than learning semantic features.

**Solution**: Random ±20 pixel displacement from the best patch location

```python
shift_x = np.random.randint(-20, 20)
shift_y = np.random.randint(-20, 20)
final_patch = frame[best_x + shift_x : best_x + shift_x + 160,
                    best_y + shift_y : best_y + shift_y + 160]
```

**Constraint**: Jitter distance (20 pixels) must be less than margin (30 pixels) to ensure patches remain within valid frame regions.

### 2.5 Channel Splitting

**Problem**: Color consistency between frames provides a low-level shortcut for ordering—the network might learn to match color histograms rather than semantic content.

**Solution**: For each frame, randomly select one RGB channel and duplicate it across all three channels.

```python
rgb = random.randint(0, 2)
frame = frame[:, :, rgb]
frame = np.stack((frame,)*3, axis=2)  # Pseudo-grayscale
```

**Implementation Status**: Channel splitting is implemented but currently disabled (commented out) in the preprocessing pipeline. Initial experiments showed training instability when combined with other augmentations. This represents a deviation from the original paper that warrants further investigation.

### 2.6 Frame Decimation

**Problem**: UCF-101 videos at 25fps contain high temporal redundancy—consecutive frames are nearly identical.

**Solution**: Keep every 2nd frame, effectively reducing to ~12.5fps.

```python
frames = frames[::2]  # Drop every other frame
```

**Rationale**: This matches the original paper's approach and increases temporal separation between selectable frames, making the ordering task more meaningful.

### 2.7 Horizontal Mirroring

**Implementation**: 50% probability of flipping all selected frames horizontally.

```python
if random.randint(0, 1) == 1:
    selected_frames = [np.flip(frame, axis=1) for frame in selected_frames]
```

**Rationale**: Standard data augmentation that effectively doubles the dataset size without changing temporal relationships.

---

## 3. Label Generation: Frame Ordering Classes

### 3.1 Permutation Reduction

For 4 frames, there are 4! = 24 possible orderings. However, many actions are visually similar forward and backward (e.g., swinging a golf club, opening/closing a door). The original paper groups forward and backward permutations into the same class:

```
Original: (0,1,2,3) and (3,2,1,0) → Same class
```

This reduces 24 permutations to 12 canonical classes.

### 3.2 Canonical Form Computation

```python
def get_frame_order_label(order_indices):
    # Canonical form: first element < last element
    if order_indices[0] < order_indices[-1]:
        canonical = order_indices
    else:
        canonical = order_indices[::-1]

    return label_dict[tuple(canonical)]
```

### 3.3 Label Mapping

| Class | Canonical Order | Class | Canonical Order |
|-------|----------------|-------|----------------|
| 0 | (0,1,2,3) | 6 | (1,0,2,3) |
| 1 | (0,2,1,3) | 7 | (1,0,3,2) |
| 2 | (0,3,2,1) | 8 | (1,2,0,3) |
| 3 | (0,1,3,2) | 9 | (1,3,0,2) |
| 4 | (0,3,1,2) | 10 | (2,0,1,3) |
| 5 | (0,2,3,1) | 11 | (2,1,0,3) |

---

## 4. Memory Management and Scalability

### 4.1 The Memory Problem

Initial naive implementation attempted to:
1. Load all UCF-101 videos into memory
2. Preprocess all frames
3. Store all preprocessed tuples

**Result**: Out-of-memory crashes. UCF-101 contains 13,320 videos; loading even 10% exceeded available RAM in the development environment.

### 4.2 Solution: Batched Preprocessing with Cloud Storage

**Architecture**:
```
Azure Blob Storage (Raw Videos)
        ↓
    Generator-based loading (yield one video at a time)
        ↓
    Batch accumulation (10 videos per batch)
        ↓
    Full preprocessing (4-frame tuples extracted)
        ↓
    Serialize to .pth file
        ↓
Azure Blob Storage (Preprocessed batches)
```

**Key Design Decisions**:

1. **Generator Pattern**: Videos are loaded one-at-a-time via Python generators, never holding more than one video in memory during the loading phase.

2. **Batch Size = 10**: Empirically determined to balance memory usage with preprocessing efficiency. Each batch produces ~10 samples (one per video) serialized together.

3. **Full Pre-extraction**: Unlike the original implementation which performed preprocessing on-the-fly during training, this implementation pre-extracts all 4-frame tuples and stores them. This trades storage space for training speed.

4. **Pickle Protocol 5**: PyTorch's `torch.save` uses pickle; protocol 5 (introduced in Python 3.8) handles large objects more efficiently.

### 4.3 Chunked Training

**Architecture**:
```
Azure Blob Storage (Preprocessed .pth files)
        ↓
    List all .pth files → 80/20 train/val split
        ↓
    For each epoch:
        Shuffle file list
        For each chunk of files:
            Load chunk into memory
            Create DataLoader
            Train on chunk
            Delete from memory
            gc.collect()
```

**Key Parameters**:
- `chunk_size = 1`: Load one .pth file at a time (conservative memory usage)
- `batch_size = 32`: Mini-batch size within each chunk
- Files shuffled each epoch for training randomness

### 4.4 Local Caching (Latest Addition)

**Problem**: Repeatedly downloading .pth files from Azure Blob Storage is slow (network latency).

**Solution**: Smart local caching with 18GB limit

```python
class FullyExtractedBlobDataset:
    MAX_CACHE_SIZE_GB = 18

    def _load_pth_files(self):
        if os.path.exists(cache_path):
            # Load from local cache
            data = torch.load(cache_path)
        else:
            # Download from Azure
            data = blob_client.download_blob().readall()
            # Cache if space available
            if current_cache_size + file_size <= MAX_CACHE_SIZE_GB:
                save_to_cache(data)
```

**Impact**: ~10x speedup on subsequent training runs (cache hits vs. network downloads).

---

## 5. Training Configuration

### 5.1 Optimizer Selection

**Original Paper**: SGD with momentum 0.9

**This Implementation**: Adam optimizer

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0003,
    betas=(0.9, 0.999),
    weight_decay=0.0005
)
```

**Rationale**: Adam's adaptive learning rates typically provide faster convergence and more stable training, especially when computational resources are limited. The learning rate (0.0003) is lower than typical Adam defaults (0.001) to compensate for Adam's tendency toward larger effective step sizes compared to SGD.

### 5.2 Learning Rate Schedule

```python
scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
```

Learning rate reduced by 10x at epochs 10 and 20, following the original paper's step-decay strategy.

### 5.3 Loss Function

Standard cross-entropy loss for 12-class classification:

```python
criterion = nn.CrossEntropyLoss()
```

### 5.4 Training Duration

- **Current configuration**: 30 epochs
- **Original paper**: ~200k iterations (approximately equivalent given batch sizes)

The reduced epoch count reflects resource constraints; validation accuracy curves are monitored for convergence.

---

## 6. Infrastructure Decisions

### 6.1 Development Environment

**Choice**: GitHub Codespaces with Docker containerization

**Rationale**:
- Reproducible environment across machines
- Pre-configured GPU access (when available)
- Integrated version control
- Consistent Python/CUDA versions

**Configuration** (`.devcontainer/devcontainer.json`):
- Base image: Python 3.10
- Key dependencies: PyTorch 2.1.1, torchvision 0.16.1, OpenCV 4.8.1

### 6.2 Data Storage

**Choice**: Microsoft Azure Blob Storage

**Rationale**:
- Student credits availability
- Python SDK with streaming support
- Scalable to BDD100K dataset (100K videos, ~1.5TB)
- Cost-effective for infrequent access patterns

**Structure**:
```
Azure Container/
├── UCF-101/                    # Raw videos by action class
│   ├── ApplyEyeMakeup/
│   ├── ApplyLipstick/
│   └── ...
└── ucf-preprocessed-data-500/  # Preprocessed batches
    ├── ucf101_preprocessed_fullyextracted_batch_1.pth
    ├── ucf101_preprocessed_fullyextracted_batch_2.pth
    └── ...
```

### 6.3 Configuration Management

**Choice**: JSON configuration file with environment variables for secrets

```json
{
    "data_prep": {
        "preprocessed_folder": "ucf-preprocessed-data-500",
        "batch_size": 10,
        "folder_limit": 101
    },
    "main": {
        "training": {
            "epochs": 30,
            "chunk_size": 1,
            "batch_size": 32
        }
    }
}
```

Credentials managed via `.env` file (not committed to version control).

---

## 7. Validation Pipeline

### 7.1 Train/Validation Split

```python
train_blob_names, val_blob_names = train_test_split(
    all_blob_names,
    test_size=0.2,
    random_state=42
)
```

80/20 split at the file level (not sample level), ensuring complete videos remain in either train or validation set.

### 7.2 Metrics Tracked

- **Training Loss**: Cross-entropy loss averaged over all training samples
- **Training Accuracy**: Fraction of correctly predicted frame orderings
- **Validation Loss**: Same as training, computed on held-out set
- **Validation Accuracy**: Primary metric for model selection

### 7.3 Expected Performance

The original OPN paper reports ~60% accuracy on the frame ordering task (random baseline: 8.3% for 12 classes). This implementation targets similar performance as validation of correct reproduction.

---

## 8. Fine-tuning Pipeline (Transfer Learning)

### 8.1 Purpose

The ultimate goal of self-supervised pretraining is to learn transferable visual representations. The fine-tuning pipeline evaluates this by:

1. Taking the pretrained COPN feature extractor (conv layers + FC6)
2. Replacing the order prediction head with a task-specific head
3. Fine-tuning on PASCAL VOC object detection

### 8.2 Implementation

Two scripts enable controlled comparison:
- `finetune_copn_pascal.py`: Uses pretrained COPN weights
- `finetune_copn_pascal-no-pretraining.py`: Random initialization (baseline)

### 8.3 Evaluation Metric

Mean Average Precision (mAP) on PASCAL VOC validation set, following standard object detection evaluation protocols.

---

## 9. Development Timeline

| Date | Milestone | Key Decisions |
|------|-----------|---------------|
| Nov 2023 | Project initialization | Selected OPN as base method |
| Nov 2023 | First training model draft | PyTorch over TensorFlow |
| Dec 2023 | Spatial jittering & patch selection | Farneback optical flow |
| Dec 2023 | Channel splitting implementation | Later disabled due to instability |
| Jun 2024 | Config file & validation set | Separated concerns |
| Jun 2024 | Pascal fine-tuning draft | Transfer learning validation |
| Jul 2024 | Docker containerization | Reproducibility priority |
| Sep 2024 | Azure Blob Storage integration | Scalability for BDD100K |
| Nov 2024 | Batched preprocessing | Solved memory crashes |
| Jan 2025 | Full pre-extraction pipeline | Trade storage for speed |
| Jan 2026 | Local caching implementation | 10x training speedup |

---

## 10. Known Limitations and Future Work

### 10.1 Current Limitations

1. **Channel Splitting Disabled**: Needs investigation for training stability
2. **No GPU Utilization**: Current training runs on CPU (Codespaces limitation)
3. **Single Dataset**: UCF-101 only; BDD100K integration pending

### 10.2 Deviations from Original Paper

| Aspect | Original OPN | This Implementation |
|--------|--------------|---------------------|
| Framework | Caffe | PyTorch |
| Optimizer | SGD + Momentum | Adam |
| Preprocessing | On-the-fly | Full pre-extraction |
| Channel Splitting | Enabled | Disabled |
| Training Iterations | 200k | ~30 epochs |

### 10.3 BDD100K Integration (Implemented)

The BDD100K integration adapts the validated UCF-101 pipeline to the Berkeley DeepDrive 100K autonomous driving dataset. The UCF-101 pipeline is preserved as-is for baseline comparison.

#### 10.3.1 BDD100K Dataset Characteristics

| Property | UCF-101 | BDD100K | Implication |
|----------|---------|---------|-------------|
| Resolution | 320×240 | 1280×720 (720p) | Downscale before processing |
| Frame Rate | 25 FPS | 30 FPS | Higher frame decimation needed |
| Duration | ~7 sec avg | ~40 sec | 6.6× more frames per video |
| Format | `.avi` (DivX) | `.mov` (H.264) | Different codec handling |
| Directory | Class-based folders | Flat `train/val/test` | No action class labels |
| Total Videos | 13,320 | 100,000 | 7.5× more data |
| Domain | Human actions | Driving scenes | Different motion patterns |

#### 10.3.2 Preprocessing Adaptations

The BDD100K pipeline (`data_prep_bdd100k.py`) makes the following adjustments:

1. **Frame Decimation**: Every 3rd frame (30fps → ~10fps) vs every 2nd frame for UCF-101. Driving videos at 30fps have even higher temporal redundancy than action videos.

2. **Resolution Downscaling**: Frames are downscaled to 640×360 before patch selection and optical flow computation. This is necessary because:
   - 720p optical flow computation is ~16× slower than 320×240
   - The 160×160 patch extraction at 720p would capture a very small portion of the frame
   - 640×360 maintains sufficient detail while being computationally tractable

3. **Optical Flow Downscaling**: Further downscaled to 320×180 for the Farneback optical flow used in frame selection weights (separate from the patch selection flow).

4. **Margin**: Increased from 30px to 60px to account for dashcam lens distortion at frame edges and the higher resolution working space (640×360).

5. **Spatial Jitter**: Increased from ±20px to ±40px proportional to the larger working resolution. The constraint `jitter < margin` is maintained (40 < 60).

6. **Video Reading**: Handles `.mov` format with rotation metadata correction. Many BDD100K dashcam videos contain `rotate:270` metadata from smartphone recording orientation.

7. **No Action Labels**: BDD100K videos don't have action class labels. The `action_label` field in the 8-tuple output is set to -1. The self-supervised frame ordering task is label-agnostic.

#### 10.3.3 Architecture

The model architecture (`model.py`) requires **no changes**. The COPN model processes 4×160×160 patches regardless of the source video's original resolution or domain. This is a key advantage of the patch-based approach.

#### 10.3.4 Training Pipeline

`bdd100k_main.py` mirrors `main.py` with BDD100K-specific configuration:
- Reads from `bdd100k-preprocessed-data/` folder in Azure
- Uses separate local cache directory (`local_cache_bdd100k/`)
- Saves model snapshots to `saved/bdd100k_modelsnapshots/`
- Same optimizer (Adam, lr=0.0003) and scheduler (MultiStepLR) for fair comparison

#### 10.3.5 Expected Differences from UCF-101

- **Motion patterns**: Driving scenes have more predictable motion (forward movement, lane changes) compared to diverse human actions. The frame ordering task may be easier or harder depending on how distinctive temporal cues are in driving.
- **Static backgrounds**: Highway driving may have large static sky/road regions, making motion-aware patch selection particularly important.
- **Scale**: 100K videos vs 13K provides significantly more training data, which may improve learned representations despite the narrower domain.

#### 10.3.6 Execution Order

```
1. python data_prep_bdd100k.py    # Preprocess BDD100K videos → .pth files in Azure
2. python bdd100k_main.py         # Train COPN on BDD100K preprocessed data
```

### 10.4 Remaining Future Work

1. **Channel Splitting Stabilization**: Debug training instability when enabled
2. **GPU Training**: Migrate to GPU-enabled environment for faster iteration
3. **Cross-domain Transfer**: Compare UCF-101 pretrained vs BDD100K pretrained representations
4. **BDD100K Fine-tuning Evaluation**: Adapt PASCAL VOC fine-tuning scripts or evaluate on BDD100K detection task

---

## 11. File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `model.py` | 78 | COPN architecture definition |
| `data_prep.py` | 683 | UCF-101 preprocessing pipeline + Azure integration |
| `data_prep_bdd100k.py` | ~350 | BDD100K preprocessing pipeline + Azure integration |
| `main.py` | 397 | UCF-101 training loop + validation |
| `bdd100k_main.py` | ~280 | BDD100K training loop + validation |
| `finetune_copn_pascal.py` | 269 | Transfer learning with pretraining |
| `finetune_copn_pascal-no-pretraining.py` | 252 | Transfer learning baseline |
| `config.json` | ~50 | Configuration parameters (UCF-101 + BDD100K) |
| `test_caching.py` | 93 | Cache performance validation |
| **Total** | **~2,450** | — |

---

## 12. Reproducibility

All code is available at: `github.com/erikjsun/OPN-masterthesis`

### UCF-101 Pipeline
1. Clone repository
2. Configure `.env` with Azure credentials
3. Run `python data_prep.py` (preprocessing)
4. Run `python main.py` (training)

### BDD100K Pipeline
1. Upload BDD100K videos to Azure Blob Storage under `bdd100k/videos/{train,val,test}/`
2. Configure `.env` with Azure credentials
3. Run `python data_prep_bdd100k.py` (preprocessing)
4. Run `python bdd100k_main.py` (training)

Environment: Python 3.10, PyTorch 2.1.1, see `requirements.txt` for full dependencies.

---

*Document updated February 2026 — BDD100K integration added*
