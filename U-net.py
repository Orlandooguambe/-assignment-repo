# =========================
# Colab: U-Net TGS Salt — FULL CODE
# =========================

# --------- 0) Setup ---------
!pip -q install -U tensorflow==2.15.0 albumentations==1.4.15 opencv-python-headless==4.9.0.80

import os, glob, random, zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

tf.random.set_seed(0)
np.random.seed(0)
random.seed(0)

print("TF:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

# --------- 1A) Kaggle API (recommended) ---------
# Run THIS cell if you want to use Kaggle API (you need to upload kaggle.json)
# After creating the token in Kaggle website (Account > Create API Token).
USE_KAGGLE = True  # set False if you prefer manual upload

if USE_KAGGLE:
    from google.colab import files
    print("Upload your kaggle.json")
    up = files.upload()
    assert 'kaggle.json' in up, "Please upload kaggle.json"
    !mkdir -p ~/.kaggle
    !cp kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json
    !kaggle competitions download -c tgs-salt-identification-challenge -p /content/tgs_salt
    !unzip -qo /content/tgs_salt/train.zip -d /content/tgs_salt
    !unzip -qo /content/tgs_salt/train_masks.zip -d /content/tgs_salt
    !unzip -qo /content/tgs_salt/test.zip -d /content/tgs_salt
    !unzip -qo /content/tgs_salt/depths.csv.zip -d /content/tgs_salt
    !unzip -qo /content/tgs_salt/sample_submission.csv.zip -d /content/tgs_salt

# --------- 1B) Manual upload (alternative) ---------
# If NOT using Kaggle API, zip locally 'tgs_salt/train/images/*.png' and 'tgs_salt/train/masks/*.png'
# and (optional) 'tgs_salt/test/images/*.png' and upload here.
if not USE_KAGGLE:
    from google.colab import files
    print("Upload the zip file containing folder tgs_salt/")
    up2 = files.upload()
    zipname = next(iter(up2))
    with zipfile.ZipFile(zipname, 'r') as zf:
        zf.extractall('/content')

DATA_DIR = "/content/tgs_salt"
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "images")
TRAIN_MASK_DIR = os.path.join(DATA_DIR, "train", "masks")
print("Training examples:", len(glob.glob(os.path.join(TRAIN_IMG_DIR, "*.png"))))

# --------- 2) Split + tf.data with albumentations ---------
!pip -q install -U albumentations==1.4.15
import albumentations as A
from sklearn.model_selection import train_test_split

IMG_SIZE = 128
img_paths = sorted(glob.glob(os.path.join(TRAIN_IMG_DIR, "*.png")))
mask_paths = [p.replace("/images/", "/masks/") for p in img_paths]

train_img, val_img, train_mask, val_mask = train_test_split(
    img_paths, mask_paths, test_size=0.15, random_state=0)

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
    img = (img.astype(np.float32) / 255.0)[..., None]
    msk = ((msk > 127).astype(np.float32))[..., None]
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

BATCH = 16
ds_tr = make_tfds(train_img, train_mask, batch=BATCH, shuffle=True, augment=True)
ds_va = make_tfds(val_img,   val_mask,   batch=BATCH, shuffle=False, augment=False)

# --------- 3) Metrics, loss and U-Net ---------
def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    inter = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.*inter + smooth)/(tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + (1.0 - dice_coef(y_true, y_pred))

def iou_metric(y_true, y_pred, thresh=0.5, eps=1e-7):
    y_pred_ = tf.cast(y_pred > thresh, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred_)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_) - inter + eps
    return inter/union

from tensorflow.keras import layers, models

def conv_block(x, filters, k=3, act='relu'):
    x = layers.Conv2D(filters, k, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(act)(x)
    x = layers.Conv2D(filters, k, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(act)(x)
    return x

def build_unet(img_size=128, ch=1, base=32):
    inputs = layers.Input((img_size, img_size, ch))
    c1 = conv_block(inputs, base)
    p1 = layers.MaxPool2D(2)(c1)
    c2 = conv_block(p1, base*2)
    p2 = layers.MaxPool2D(2)(c2)
    c3 = conv_block(p2, base*4)
    p3 = layers.MaxPool2D(2)(c3)
    c4 = conv_block(p3, base*8)
    p4 = layers.MaxPool2D(2)(c4)
    bn = conv_block(p4, base*16)
    u5 = layers.UpSampling2D(2)(bn); u5 = layers.Concatenate()([u5, c4]); c5 = conv_block(u5, base*8)
    u6 = layers.UpSampling2D(2)(c5); u6 = layers.Concatenate()([u6, c3]); c6 = conv_block(u6, base*4)
    u7 = layers.UpSampling2D(2)(c6); u7 = layers.Concatenate()([u7, c2]); c7 = conv_block(u7, base*2)
    u8 = layers.UpSampling2D(2)(c7); u8 = layers.Concatenate()([u8, c1]); c8 = conv_block(u8, base)
    out = layers.Conv2D(1, 1, activation='sigmoid')(c8)
    return models.Model(inputs, out)

model = build_unet(IMG_SIZE, 1, base=32)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=bce_dice_loss,
              metrics=[dice_coef, iou_metric])
model.summary()

# --------- 4) Training ---------
EPOCHS = 15
ckpt = tf.keras.callbacks.ModelCheckpoint("unet_tgs.h5", monitor="val_iou_metric",
                                          save_best_only=True, mode="max", verbose=1)
early = tf.keras.callbacks.EarlyStopping(monitor="val_iou_metric", patience=5,
                                         mode="max", restore_best_weights=True)

hist = model.fit(ds_tr, validation_data=ds_va, epochs=EPOCHS,
                 callbacks=[ckpt, early], verbose=1)

# --------- 5) Curves ---------
hs = hist.history
plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.plot(hs['loss']); plt.plot(hs['val_loss']); plt.title('loss'); plt.grid(); plt.legend(['train','val'])
plt.subplot(1,3,2); plt.plot(hs['dice_coef']); plt.plot(hs['val_dice_coef']); plt.title('dice'); plt.grid(); plt.legend(['train','val'])
plt.subplot(1,3,3); plt.plot(hs['iou_metric']); plt.plot(hs['val_iou_metric']); plt.title('IoU'); plt.grid(); plt.legend(['train','val'])
plt.tight_layout(); plt.show()

# --------- 6) Prediction visualization ---------
batch = next(iter(ds_va))
imgs, gts = batch
preds = model.predict(imgs, verbose=0)

n = 6
plt.figure(figsize=(12,6))
for i in range(n):
    plt.subplot(3,n,i+1);        plt.imshow(imgs[i,...,0], cmap='gray'); plt.axis('off'); plt.title('Image')
    plt.subplot(3,n,i+1+n);      plt.imshow(gts[i,...,0], cmap='gray');  plt.axis('off'); plt.title('GT')
    plt.subplot(3,n,i+1+2*n);    plt.imshow((preds[i,...,0]>0.5).astype(np.uint8), cmap='gray'); plt.axis('off'); plt.title('Pred')
plt.tight_layout(); plt.show()

# --------- 7) (Optional) RLE submission generation ---------
TEST_IMG_DIR = os.path.join(DATA_DIR, "test", "images")
test_img_paths = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.png")))
print("Test imgs:", len(test_img_paths))

def rle_encode(mask):
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

ids, rles = [], []
limit = min(500, len(test_img_paths))  # demo limit
for p in test_img_paths[:limit]:
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    im = (im.astype(np.float32)/255.0)[...,None]
    pr = model.predict(im[None,...], verbose=0)[0,...,0]
    pr_bin = (pr>0.5).astype(np.uint8)
    pr_101 = cv2.resize(pr_bin, (101,101), interpolation=cv2.INTER_NEAREST)
    img_id = os.path.splitext(os.path.basename(p))[0]
    ids.append(img_id)
    rles.append(rle_encode(pr_101))

if len(ids):
    sub = pd.DataFrame({"id": ids, "rle_mask": rles})
    sub.to_csv("submission.csv", index=False)
    print(sub.head())
    from google.colab import files as colab_files
    # colab_files.download("submission.csv")  # uncomment to download automatically