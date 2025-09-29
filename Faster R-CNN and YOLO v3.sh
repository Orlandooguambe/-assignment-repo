# ===============================
# ALL-IN-ONE: Faster R-CNN + YOLOv3
# ===============================
# Colab notebook style script

# -------------------------------
# SECTION A. YOLOv3 (Keras, TF2.x)
# -------------------------------
!git clone https://github.com/qqwweee/keras-yolo3.git
%cd keras-yolo3

# Setup versions that work
!pip -q uninstall -y keras-nightly keras tf-keras-nightly > /dev/null 2>&1
!pip -q install "tensorflow==2.17.1" "tf-keras==2.17.0" "numpy==1.26.4" "h5py==3.10.0" \
                "opencv-python-headless==4.9.0.80" "pillow" "matplotlib" "tqdm"
import tensorflow as tf, tf_keras as keras
print("YOLOv3 section — TF:", tf.__version__, "| tf_keras:", keras.__version__)

# Download pretrained YOLOv3 weights and convert
!wget -q https://pjreddie.com/media/files/yolov3.weights -O yolov3.weights
!python convert.py yolov3.cfg yolov3.weights model_data/yolov3.h5

# Run inference on sample images
!mkdir -p imgs
!wget -q https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg -O imgs/street.jpg
!wget -q https://images.pexels.com/photos/399610/pexels-photo-399610.jpeg -O imgs/dog.jpg

!python yolo_video.py --model model_data/yolov3.h5 --classes model_data/coco_classes.txt \
                      --image --input imgs/street.jpg --output out_street.jpg
!python yolo_video.py --model model_data/yolov3.h5 --classes model_data/coco_classes.txt \
                      --image --input imgs/dog.jpg --output out_dog.jpg

from IPython.display import Image, display
display(Image("out_street.jpg"))
display(Image("out_dog.jpg"))

# (Optional) Training on Simpsons dataset:
# Prepare "train.txt" and "simpsons_classes.txt" in model_data/ then run:
# !python train.py --annotation_path train.txt \
#                  --classes_path model_data/simpsons_classes.txt \
#                  --anchors_path model_data/yolo_anchors.txt \
#                  --batch_size 8 --epochs 30 \
#                  --weights_path model_data/yolov3.h5 \
#                  --log_dir logs/simpsons_yolov3


# -------------------------------
# SECTION B. Faster R-CNN (Keras 2.3, TF1.x)
# -------------------------------
%cd /content
!pip -q install "tensorflow==1.15.5" "keras==2.3.1" "opencv-python-headless==4.5.5.64" \
                "pillow==8.4.0" "h5py==2.10.0" "numpy==1.19.5" "imgaug==0.4.0"
import tensorflow as tf, keras
print("Faster R-CNN section — TF:", tf.__version__, "| Keras:", keras.__version__)

# Clone repo
!git clone https://github.com/kentaroy47/keras-frcnn.git
%cd keras-frcnn

# Download pretrained weights
!wget -q https://github.com/kentaroy47/keras-frcnn/releases/download/v0.1/frcnn_vgg.hdf5 -O model_frcnn.hdf5

# Run prediction on current dir
!python test_frcnn.py --path ./ --config ./config.pickle --network vgg \
        --input_weight_path model_frcnn.hdf5 --num_rois 32 --output test_out

import glob
outs = glob.glob("test_out/*.jpg")[:3]
outs
