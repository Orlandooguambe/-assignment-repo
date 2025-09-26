# =========================
# Colab: TGS Salt — U-Net (ResNet vs VGG) — FULL PIPELINE
# =========================

# --------- 0) Environment setup ---------
# Stable combo for segmentation_models with tf.keras backend
!pip -q install -U \
  "tensorflow==2.15.0" "tf-keras==2.15.0" \
  "segmentation-models==1.0.1" \
  "albumentations==1.4.15" \
  "opencv-python-headless==4.9.0.80" \
  "numpy==1.26.4" "pandas" "matplotlib" "scikit-learn"

import os, glob, random, zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

import segmentation_models as sm
sm.set_framework('tf.keras'); sm.framework()

tf.random.set_seed(0)
np.random.seed(0)
random.seed(0)

print("TF:", tf.__version__)
print("tf-keras:", tf.keras.__version__)
print("segmentation_models:", sm.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

# --------- 1A) Kaggle API (recommended) ---------
# Set USE_KAGGLE=True to download the competition data automatically.
USE_KAGGLE = True  # set False to upload your own prepared zip

DATA_DIR = "/content/tgs_salt"
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "images")
TRAIN_MASK_DIR = os.path.join(DATA_DIR, "train", "masks")
TEST_IMG_DIR  = os.path.join(DATA_DIR, "test", "images")

if USE_KAGGLE:
    from google.colab import files
    print("Upload your kaggle.json (https://www.kaggle.com -> Account -> Create API Token)")
    up = files.upload()
    assert 'kaggle.json' in up, "kaggle.json is required for Kaggle API"
    !mkdir -p ~/.kaggle
    !cp kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json

    !kaggle competitions download -c tgs-salt-identification-challenge -p {DATA_DIR}
    !unzip -qo {DATA_DIR}/train.zip -d {DATA_DIR}
    !unzip -qo {DATA_DIR}/train_masks.zip -d {DATA_DIR}
    !unzip -qo {DATA_DIR}/test.zip -d {DATA_DIR}
    !unzip -qo {DATA_DIR}/depths.csv.zip -d {DATA_DIR}
    !unzip -qo {DATA_DIR}/sample_submission.csv.zip -d {DATA_DIR}

# --------- 1B) Manual upload (alternative) ---------
# If not using Kaggle API: upload a zip that contains:
# tgs_salt/train/images/*.png  and  tgs_salt/train/masks/*.png
# (optional) tgs_salt/test/images/*.png
if not USE_KAGGLE:
    from google.colab import files
    print("Upload a zip containing the folder `tgs_salt/` with train/images & train/masks")
    up2 = files.upload()
    zipname = next(iter(up2))
    with zipfile.ZipFile(zipname, 'r') as zf:
        zf.extractall('/content')

print("Train images:", len(glob.glob(os.path.join(TRAIN_IMG_DIR, "*.png"))))
print("Train masks :", len(glob.glob(os.path.join(TRAIN_MASK_DIR, "*.png"))))

# --------- 2) tf.data pipeline + Albumentations ---------
import albumentations as A
from sklearn.model_selection import train_test_split

IMG_SIZE = 128  # resize target (H=W=128)
BATCH    = 16

img_paths  = sorted(glob.glob(os.path.join(TRAIN_IMG_DIR, "*.png")))
mask_paths = [p.replace("/images/", "/masks/") for p in img_paths]

# Split train/val
train_img, val_img, train_mask, val_mask = train_test_split(
    img_paths, mask_paths, test_size=0.15, random_state=0
)

train_aug = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
])

def load_pair(img_path, mask_path, augment=False):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    msk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    msk = cv2.resize(msk, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    if augment:
        out = train_aug(image=img, mask=msk)
        img, msk = out['image'], out['mask']
    img = (img.astype(np.float32) / 255.0)[..., None]         # (H,W,1)
    msk = ((msk > 127).astype(np.float32))[..., None]         # binary mask (H,W,1)
    return img, msk

def gen(paths_img, paths_msk, augment=False):
    for ip, mp in zip(paths_img, paths_msk):
        yield load_pair(ip, mp, augment)

def make_tfds(img_list, mask_list, batch=16, shuffle=True, augment=False):
    ds = tf.data.Dataset.from_generator(
        lambda: gen(img_list, mask_list, augment),
        output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 1), dtype=tf.float32),
        )
    )
    if shuffle:
        ds = ds.shuffle(2048, reshuffle_each_iteration=True)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

ds_tr = make_tfds(train_img, train_mask, batch=BATCH, shuffle=True,  augment=True)
ds_va = make_tfds(val_img,   val_mask,   batch=BATCH, shuffle=False, augment=False)

# --------- 3) Losses, metrics, model builders ---------
# Using segmentation_models losses/metrics for convenience
dice_loss = sm.losses.DiceLoss()
bce_loss  = tf.keras.losses.BinaryCrossentropy()
total_loss = dice_loss + bce_loss

metrics = [
    sm.metrics.IOUScore(threshold=0.5, name="iou"),
    sm.metrics.FScore(threshold=0.5, name="f1")
]

def build_unet(backbone_name="resnet34", input_channels=1, img_size=128):
    # If using grayscale, we can repeat channels to fit ImageNet encoders (which expect 3 channels)
    inputs = tf.keras.layers.Input((img_size, img_size, input_channels))
    if input_channels == 1:
        x = tf.keras.layers.Concatenate()([inputs, inputs, inputs])  # (H,W,3)
    else:
        x = inputs

    # Build U-Net with chosen encoder
    base = sm.Unet(
        backbone_name=backbone_name,
        encoder_weights="imagenet",
        input_shape=(img_size, img_size, 3),
        classes=1,
        activation="sigmoid"
    )
    outputs = base(x)
    model = tf.keras.Model(inputs, outputs, name=f"unet_{backbone_name}")
    return model

def freeze_encoder(model):
    # Freeze all layers belonging to the encoder (by name convention)
    for layer in model.layers:
        if "backbone" in layer.name or "encoder" in layer.name:
            layer.trainable = False
    # Some SM versions expose encoder layers via model.get_layer('backbone') etc.
    # We additionally ensure BatchNorm layers are not updating if frozen:
    for layer in model.layers:
        if not layer.trainable and isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

def unfreeze_all(model):
    for layer in model.layers:
        layer.trainable = True

# --------- 4) Build both models (ResNet34 & VGG16) ---------
resnet_model = build_unet(backbone_name="resnet34", input_channels=1, img_size=IMG_SIZE)
vgg_model    = build_unet(backbone_name="vgg16",    input_channels=1, img_size=IMG_SIZE)

# --------- 5) Training schedule (freeze -> unfreeze/fine-tune) ---------
def train_model(model, name="model", ds_tr=None, ds_va=None,
                lr_frozen=1e-3, lr_ft=5e-4, ep_frozen=8, ep_ft=12):
    print(f"\n=== {name}: freeze encoder & train decoder ===")
    freeze_encoder(model)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr_frozen), loss=total_loss, metrics=metrics)

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        f"{name}_best.h5", monitor="val_iou", mode="max",
        save_best_only=True, verbose=1
    )
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_iou", mode="max", patience=5, restore_best_weights=True, verbose=1
    )

    hist1 = model.fit(
        ds_tr, validation_data=ds_va,
        epochs=ep_frozen, callbacks=[ckpt, early], verbose=1
    )

    print(f"\n=== {name}: unfreeze & fine-tune all layers ===")
    unfreeze_all(model)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr_ft), loss=total_loss, metrics=metrics)

    ckpt2 = tf.keras.callbacks.ModelCheckpoint(
        f"{name}_ft_best.h5", monitor="val_iou", mode="max",
        save_best_only=True, verbose=1
    )
    early2 = tf.keras.callbacks.EarlyStopping(
        monitor="val_iou", mode="max", patience=6, restore_best_weights=True, verbose=1
    )

    hist2 = model.fit(
        ds_tr, validation_data=ds_va,
        epochs=ep_ft, callbacks=[ckpt2, early2], verbose=1
    )

    # merge histories
    hist = {}
    for k, v in hist1.history.items():
        hist[k] = v + hist2.history.get(k, [])
    for k, v in hist2.history.items():
        if k not in hist:
            hist[k] = hist1.history.get(k, []) + v
    return hist

# Keep epochs modest; increase after verifying the pipeline
resnet_hist = train_model(
    resnet_model, name="unet_resnet34",
    ds_tr=ds_tr, ds_va=ds_va,
    lr_frozen=1e-3, lr_ft=5e-4, ep_frozen=6, ep_ft=8
)
vgg_hist = train_model(
    vgg_model, name="unet_vgg16",
    ds_tr=ds_tr, ds_va=ds_va,
    lr_frozen=1e-3, lr_ft=5e-4, ep_frozen=6, ep_ft=8
)

# --------- 6) Plot training curves ---------
def plot_histories(histA, histB, labelA="ResNet34", labelB="VGG16"):
    keys = ["loss", "val_loss", "iou", "val_iou", "f1", "val_f1"]
    plt.figure(figsize=(12,8))
    for i, (k1, k2) in enumerate([("loss","val_loss"), ("iou","val_iou"), ("f1","val_f1")], 1):
        plt.subplot(2,3,i)
        plt.plot(histA[k1], label=f"{labelA} {k1}")
        plt.plot(histA[k2], label=f"{labelA} {k2}")
        plt.plot(histB[k1], label=f"{labelB} {k1}", linestyle="--")
        plt.plot(histB[k2], label=f"{labelB} {k2}", linestyle="--")
        plt.title(k1.replace("val_","").upper()); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.show()

plot_histories(resnet_hist, vgg_hist, "ResNet34", "VGG16")

# Print final metrics (last epoch seen)
def last_val(metrics_hist, key):
    return metrics_hist[key][-1] if key in metrics_hist and len(metrics_hist[key]) else float("nan")

print("\n=== Final Validation (last epoch logged) ===")
print(f"ResNet34  -> val_iou: {last_val(resnet_hist,'val_iou'):.4f}, val_f1: {last_val(resnet_hist,'val_f1'):.4f}")
print(f"VGG16     -> val_iou: {last_val(vgg_hist,'val_iou'):.4f}, val_f1: {last_val(vgg_hist,'val_f1'):.4f}")

# --------- 7) Visualize predictions ---------
# Take a small batch from validation set
val_batch = next(iter(ds_va))
val_imgs, val_masks = val_batch

def show_preds(model, imgs, gts, n=6, title=""):
    preds = model.predict(imgs, verbose=0)
    plt.figure(figsize=(12,6))
    for i in range(n):
        plt.subplot(3,n,i+1);        plt.imshow(imgs[i,...,0], cmap='gray'); plt.axis('off'); plt.title('Image')
        plt.subplot(3,n,i+1+n);      plt.imshow(gts[i,...,0], cmap='gray');  plt.axis('off'); plt.title('GT')
        plt.subplot(3,n,i+1+2*n);    plt.imshow((preds[i,...,0]>0.5).astype(np.uint8), cmap='gray'); plt.axis('off'); plt.title('Pred')
    plt.suptitle(title)
    plt.tight_layout(); plt.show()

show_preds(resnet_model, val_imgs, val_masks, n=6, title="ResNet34 U-Net — Validation Predictions")
show_preds(vgg_model,   val_imgs, val_masks, n=6, title="VGG16 U-Net — Validation Predictions")

# --------- 8) (Optional) Generate RLE submission on test split (demo) ---------
def rle_encode(mask):
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

test_img_paths = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.png")))
print("Test imgs:", len(test_img_paths))

# Choose which model to use for submission (best validation IoU)
use_resnet = last_val(resnet_hist,'val_iou') >= last_val(vgg_hist,'val_iou')
best_model = resnet_model if use_resnet else vgg_model
print("Using model:", "ResNet34" if use_resnet else "VGG16")

ids, rles = [], []
limit = min(500, len(test_img_paths))  # small demo limit
for p in test_img_paths[:limit]:
    im  = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    imr = cv2.resize(im, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    imn = (imr.astype(np.float32)/255.0)[...,None]
    pr  = best_model.predict(imn[None,...], verbose=0)[0,...,0]
    prb = (pr > 0.5).astype(np.uint8)
    pr_101 = cv2.resize(prb, (101,101), interpolation=cv2.INTER_NEAREST)
    img_id = os.path.splitext(os.path.basename(p))[0]
    ids.append(img_id)
    rles.append(rle_encode(pr_101))

if len(ids):
    sub = pd.DataFrame({"id": ids, "rle_mask": rles})
    sub.to_csv("submission.csv", index=False)
    print(sub.head())
    # from google.colab import files as colab_files
    # colab_files.download("submission.csv")  # uncomment to auto-download
