# NTIRE 2026 Image Denoising (σ=50) — Team 08

## Setup

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Download Weights

Place all three files (folder link: https://drive.google.com/drive/folders/1jxCEg71H4Clrm-ERoDg9O6gP9fjbxNPB?usp=sharing ) in `model_zoo/`: 

| File | Link |
|---|---|
| `team08_ScunetFT.pth` | [Download](https://drive.google.com/file/d/1UZlX8Nd6UKDhPFU4sNPVs3Q3CjkhXcmF/view?usp=sharing) |
| `team08_XformerFT.pth` | [Download](https://drive.google.com/file/d/1UZlX8Nd6UKDhPFU4sNPVs3Q3CjkhXcmF/view?usp=sharing) |
| `team08_RestormerFT.pth` | [Download](https://drive.google.com/file/d/1UZlX8Nd6UKDhPFU4sNPVs3Q3CjkhXcmF/view?usp=sharing) |

## Run

```bash
python test_demo.py --data_dir ./data --save_dir ./results --model_id 8
```