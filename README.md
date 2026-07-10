
<h1>Beyond 2D Slices: TD-Mamba for 3D CT Segmentation of Head and Neck Space-Occupying Lesions</h1>



## 1. Repository layout

```
0_model_size_and_flops.py   # print model parameter count and FLOPs
1_train.py                  # single/multi-GPU-per-process training entry point (5-fold CV)
1_train_multiGPU.py         # DDP training template for a different model (not used for Brain-Mamba)
2_predict.py                # run inference for a trained fold and save predictions as .nrrd
3_compute_metrics.py        # Dice / IoU / HD95 over all 5 folds combined
3_compute_metrics_folds.py  # Dice / IoU / HD95 for a single fold
model_brain_mamba/          # Brain-Mamba model definition
light_training/             # training engine: data loading, preprocessing, losses, trainer, metrics
causal-conv1d/, mamba/      # vendored CUDA extensions required by mamba_ssm
data/henect204_seg/         # dataset root (see DATASET.md)
experiments/henect204/      # training logs, checkpoints, predictions, metrics (created at runtime)
note.txt                    # original Vietnamese setup notes this README is based on
```

## 2. Environment setup

```bash
conda create -n 3dhnseg python=3.10
conda activate 3dhnseg

pip install packaging ninja setuptools wheel

pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

pip install "numpy<2.0"
pip install fvcore
pip install monai==1.3.0

# Mamba CUDA extensions (vendored in this repo)
cd causal-conv1d
python setup.py install
cd ..

cd mamba
python setup.py install
cd ..
```

You will also need `SimpleITK`, `blosc2`, `scikit-learn`, `tqdm`, `batchgenerators`, and `medpy` (used by the data loader and metrics computation) — install any that are missing via `pip install SimpleITK blosc2 scikit-learn tqdm batchgenerators medpy`.

A CUDA-capable GPU is required (the model relies on `mamba_ssm`'s custom CUDA kernels).

## 3. Dataset

Training reads the preprocessed `3D-HNSeg` dataset from `./data/henect204_seg/`. See [DATASET.md](DATASET.md) for the full description of the data source, directory layout, label definitions, and preprocessing steps. In short:

- 204 head-and-neck CT cases, 4 classes (`background`, `abscess`, `tumor`, `cyst`).
- Preprocessed training tensors live in `data/henect204_seg/Plans_3d_lowres/` (Blosc2 `.b2nd` + `.pkl` properties).
- Full-resolution ground-truth masks for evaluation live in `data/henect204_seg/gt_segmentations/`.
- The 5-fold split used everywhere below is `data/henect204_seg/splits_final.json`.

If you are adapting this pipeline to a different model, update the model/loss/optimizer/scheduler declarations at:

| Component | File | Line |
|---|---|---|
| Model | `0_model_size_and_flops.py` | 8 |
| Model | `1_train.py` | 27 |
| Model | `2_predict.py` | 48 |
| Loss | `1_train.py` | 37 |
| Optimizer | `1_train.py` | 43 |
| Scheduler | `1_train.py` | 46 |
| Scheduler (engine hook) | `light_training/trainer.py` | 398 |

## 4. Sanity check: model size and FLOPs

```bash
python 0_model_size_and_flops.py
```

Note: `fvcore`'s FLOP counter does not support Mamba's custom CUDA ops, so the reported FLOPs for the `ToM_Block`/SSAM components are a theoretical lower bound rather than an exact measurement.

## 5. Training (5-fold cross-validation)

Training uses `roi_size = [96, 96, 96]` patches, batch size 2, `AdamW` (lr `1e-4`), `DiceCELoss` (no background), for up to 1000 epochs, with validation every 2 epochs. The best (by mean foreground Dice) and final-epoch checkpoints are saved per fold.

Before training, create the empty log directories for each fold (the trainer does not create them automatically):

```bash
mkdir -p ./experiments/henect204/fold_0 \
         ./experiments/henect204/fold_1 \
         ./experiments/henect204/fold_2 \
         ./experiments/henect204/fold_3 \
         ./experiments/henect204/fold_4
```

Run one fold:

```bash
python 1_train.py --fold 0 --gpu 0 2>&1 | tee ./experiments/henect204/fold_0/log.txt
```

Run all 5 folds sequentially:

```bash
python 1_train.py --fold 0 2>&1 | tee ./experiments/henect204/fold_0/log.txt && \
python 1_train.py --fold 1 2>&1 | tee ./experiments/henect204/fold_1/log.txt && \
python 1_train.py --fold 2 2>&1 | tee ./experiments/henect204/fold_2/log.txt && \
python 1_train.py --fold 3 2>&1 | tee ./experiments/henect204/fold_3/log.txt && \
python 1_train.py --fold 4 2>&1 | tee ./experiments/henect204/fold_4/log.txt
```

Useful flags:
- `--gpu <id>`: CUDA device index (default `0`).
- `--checkpoint <path>`: resume training from a saved checkpoint (`best_checkpoint.pth` / `latest_checkpoint.pth`).

Checkpoints and logs for each fold are written to `./experiments/henect204/fold_<i>/logs/model/`:
- `best_model_epoch<N>_<dice>.pt` — best model weights so far (kept, previous best deleted).
- `inference_model.pt` — a copy of the current best model, used by `2_predict.py`.
- `final_model_<dice>.pt` — weights from the most recent completed epoch.
- `best_checkpoint.pth` / `latest_checkpoint.pth` — full training state (for resuming), including `best_mean_dice`.

3D volumetric augmentation is RAM-intensive. If training crashes with `RuntimeError: One or more background workers are no longer alive.`, reduce `train_process` (dataloader worker count, default `8` in `1_train.py`).

## 6. Inference

Before running inference, create the shared prediction output directory:

```bash
mkdir -p ./experiments/henect204/prediction_results
```

Run inference for a single fold using that fold's `inference_model.pt` checkpoint:

```bash
python 2_predict.py --fold 0 --gpu 0 2>&1 | tee -a ./experiments/henect204/prediction_results/log.txt
```

Run all 5 folds:

```bash
python 2_predict.py --fold 0 2>&1 | tee -a ./experiments/henect204/prediction_results/log.txt && \
python 2_predict.py --fold 1 2>&1 | tee -a ./experiments/henect204/prediction_results/log.txt && \
python 2_predict.py --fold 2 2>&1 | tee -a ./experiments/henect204/prediction_results/log.txt && \
python 2_predict.py --fold 3 2>&1 | tee -a ./experiments/henect204/prediction_results/log.txt && \
python 2_predict.py --fold 4 2>&1 | tee -a ./experiments/henect204/prediction_results/log.txt
```

Use `--model_checkpoint_path <path>` to evaluate a specific checkpoint instead of the default `./experiments/henect204/fold_<fold>/logs/model/inference_model.pt`.

Predictions are saved as `.nrrd` volumes (resampled back to the case's original spacing/origin/direction) at:

```
./experiments/henect204/prediction_results/<case_id>.nrrd
```

## 7. Metrics computation

Metrics (**Dice**, **IoU/Jaccard**, **HD95**) are computed per class (`abscess`, `tumor`, `cyst`) against the full-resolution ground truth in `data/henect204_seg/gt_segmentations/`.

Across all 5 folds combined (204 cases):

```bash
mkdir -p ./experiments/henect204/result_metrics
python 3_compute_metrics.py 2>&1 | tee -a ./experiments/henect204/result_metrics/log.txt
```

For a single fold:

```bash
python 3_compute_metrics_folds.py --folds 0
```

(repeat with `--folds 1` … `--folds 4` for the other folds).

## 8. Model, loss, and training details

- **Model**: `Brain_Mamba` (`model_brain_mamba/Brain_Mamba.py`) — a 5-stage 3D encoder-decoder. Each stage stacks `TD_MambaBlock`s (multi-dilation depthwise convolutions + a `ToM_Block` state-space mixer), downsampled/upsampled by `TD_MambaDownBlock`/`TD_MambaUpBlock`. Skip connections pass through a memory-augmented cross-update block (`TD_Mamba_Memory_Skip_Block`) before being concatenated with the decoder's upsampled features and projected back to the working channel size.
- **Loss**: `DiceCELoss` (MONAI), one-hot targets, softmax, background excluded.
- **Optimizer**: `AdamW`, lr `1e-4`, gradient clipping at norm `1.0`.
- **Scheduler**: none (constant learning rate).
- **Validation metric**: mean foreground Dice over `abscess`/`tumor`/`cyst`, logged each validation step and used for best-checkpoint selection.

## Acknowledgement

This codebase builds on:

- [SegMamba](https://github.com/ge-xing/SegMamba) — long-range sequential modeling with Mamba for 3D medical image segmentation.
- [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) — data preprocessing/planning pipeline.
- [Project-MONAI/MONAI](https://github.com/Project-MONAI/MONAI) — losses, inferers, and transforms.
- [state-spaces/mamba](https://github.com/state-spaces/mamba) and [hustvl/Vim](https://github.com/hustvl/Vim) — Mamba/bidirectional-Mamba implementations.

## Contact / data availability

The `3D-HNSeg` dataset and the full training/inference source code will be made available through the project repository:

**https://github.com/drthaodao3101/3D-HNSeg**

For questions about the dataset or this codebase, please open an issue on that repository.
