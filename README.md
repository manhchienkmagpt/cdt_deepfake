# chien_deepfake

Project huan luyen va danh gia bai toan Deepfake Detection / Deepfake Classification theo binary classification:

- `real = 0`
- `fake = 1`

Project duoc rut gon theo cau truc va logic chinh cua DeepfakeBench cho 5 model:

- EfficientB4
- FWA
- UCF
- SRM
- SPSL

## 1. Cai moi truong

Nen dung Python 3.9+ va tao moi truong rieng:

```bash
python -m venv .venv
.venv\Scripts\activate
```

## 2. Cai requirements

```bash
pip install -r requirements.txt
```

Neu dung GPU, hay cai ban PyTorch phu hop voi CUDA tren may truoc khi cai cac package con lai.

## 3. Chuan bi dataset

Mac dinh config doc dataset FF++ tai:

```text
D:/duong_huy_ct7/deepfake-data
```

Va CelebDF test tai:

```text
D:/duong_huy_ct7/deepfake-data/celeb-df/test
```

Co the sua lai trong tung file `configs/*.yaml`.

## 4. Cau truc folder dataset

FF++:

```text
deepfake-data/
├── train/
│   ├── original/
│   ├── Deepfakes/
│   ├── Face2Face/
│   ├── FaceShifter/
│   ├── FaceSwap/
│   └── NeuralTextures/
├── val/
│   ├── original/
│   ├── Deepfakes/
│   ├── Face2Face/
│   ├── FaceShifter/
│   ├── FaceSwap/
│   └── NeuralTextures/
└── test/
    ├── original/
    ├── Deepfakes/
    ├── Face2Face/
    ├── FaceShifter/
    ├── FaceSwap/
    └── NeuralTextures/
```

CelebDF:

```text
celeb-df/
└── test/
    ├── real/
    └── fake/
```

Moi folder class chi chua anh truc tiep, khong dung subfolder video. Cac duoi anh ho tro: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.

## 5. Cach train tung model

Chay lenh trong folder `chien_deepfake`:

```bash
python train.py --model efficientb4 --config configs/efficientb4.yaml
python train.py --model fwa --config configs/fwa.yaml
python train.py --model ucf --config configs/ucf.yaml
python train.py --model srm --config configs/srm.yaml
python train.py --model spsl --config configs/spsl.yaml
```

Quy trinh train:

1. Train tren FF++ train.
2. Validate tren FF++ val.
3. Luu checkpoint tot nhat theo validation `AUC`.
4. Early stopping neu validation `AUC` khong cai thien.
5. Load checkpoint tot nhat va evaluate tren FF++ test va CelebDF test.

Checkpoint tot nhat duoc luu tai:

```text
checkpoints/<model>_best.pth
```

## 6. Cach train voi them fake GAN

Dung `train_with_gan.py` khi muon train giong `train.py`, nhung them mot folder anh fake khac vao rieng FF++ train split. Tat ca anh trong folder GAN se duoc gan label `fake = 1`.

Vi du folder GAN fake:

```text
gan-fake-images/
├── image_001.jpg
├── image_002.png
└── run_01/
    └── image_003.jpg
```

Mac dinh code se quet ca cac subfolder ben trong `gan_fake_root`. Co the chay bang CLI:

```bash
python train_with_gan.py --model efficientb4 --config configs/efficientb4.yaml --gan-fake-root "D:/path/to/gan-fake-images"
```

Thay model/config tuong ung neu can:

```bash
python train_with_gan.py --model fwa --config configs/fwa.yaml --gan-fake-root "D:/path/to/gan-fake-images"
python train_with_gan.py --model ucf --config configs/ucf.yaml --gan-fake-root "D:/path/to/gan-fake-images"
python train_with_gan.py --model srm --config configs/srm.yaml --gan-fake-root "D:/path/to/gan-fake-images"
python train_with_gan.py --model spsl --config configs/spsl.yaml --gan-fake-root "D:/path/to/gan-fake-images"
```

Hoac cau hinh truc tiep trong `configs/*.yaml`:

```yaml
data:
  gan_fake_root: "D:/path/to/gan-fake-images"
  gan_fake_recursive: true
```

Sau do chay:

```bash
python train_with_gan.py --model efficientb4 --config configs/efficientb4.yaml
```

Neu muon ket hop voi upsampling real:

```bash
python train_with_gan.py --model efficientb4 --config configs/efficientb4.yaml --gan-fake-root "D:/path/to/gan-fake-images" --real-upsample-factor 4
```

Luu y:

- GAN fake chi duoc them vao FF++ train.
- FF++ val, FF++ test va CelebDF test giu nguyen de danh gia cong bang.
- Neu khong truyen `--gan-fake-root` va `data.gan_fake_root` dang la `null`, chuong trinh se bao loi.

## 7. Cach test FF++ test

```bash
python test.py --model efficientb4 --config configs/efficientb4.yaml --checkpoint checkpoints/efficientb4_best.pth --dataset ffpp --split test
```

Thay `efficientb4` bang `fwa`, `ucf`, `srm`, hoac `spsl` neu can test model khac.

## 8. Cach test cross-dataset CelebDF

```bash
python test.py --model efficientb4 --config configs/efficientb4.yaml --checkpoint checkpoints/efficientb4_best.pth --dataset celebdf
```

## 9. Cach xem ket qua metrics

Metrics duoc in ra terminal va luu JSON trong folder `results/`, vi du:

```text
results/
├── efficientb4_ffpp_test_metrics.json
└── efficientb4_celebdf_test_metrics.json
```

Ten metric trong file JSON:

- `Accuracy`
- `F1_score`
- `Precision`
- `Recall`
- `AUC`

## 10. Mapping label real/fake

FF++:

- `original` -> real -> label `0`
- `Deepfakes`, `Face2Face`, `FaceShifter`, `FaceSwap`, `NeuralTextures` -> fake -> label `1`

CelebDF:

- `real` -> label `0`
- `fake` -> label `1`

## Ghi chu pretrained

Config mac dinh tro den:

```text
pretrained/xception-b5690688.pth
pretrained/efficientnet-b4-6ed6700e.pth
```

Neu file pretrained khong ton tai, code se log warning va train tu dau. De dung pretrained, tao folder `pretrained/` trong `chien_deepfake` va dat file weight tuong ung vao do, hoac sua duong dan trong config.

## Ghi chu UCF

UCF goc cua DeepfakeBench dung pair dataset va label rieng cho tung loai fake de tinh them specific/reconstruction/contrastive loss. Dataset moi trong project nay chi co flat image folder va binary label, nen module UCF giu dual encoder va common/specific feature heads, nhung training/evaluation dung binary common head de chay dung voi output `image, label, image_path`.
