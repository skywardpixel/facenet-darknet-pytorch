# Facenet w/ Darknet in Pytorch

**Work in progress. Course project for Computer Vision.**

A [PyTorch](https://pytorch.org/) implementation of the [Facenet](https://arxiv.org/abs/1503.03832) model for face
recognition. A port of [facenet-darknet-inference](https://github.com/lincolnhard/facenet-darknet-inference) to
PyTorch.

## Instructions

1. Download [weights](https://drive.google.com/open?id=1ATzb5ZEQo424wlSY-cdlT54FUWlIry8V) and extract.
2. Put `facenet.weights`, `haarcascade_frontalface_alt2.xml` and `shape_predictor_68_face_landmarks.dat` in
`weights/`.
3. Install dependencies using `conda` or `pip`.
4. Run `python main.py`. Type `a` to register new face, `r` to recognize face from camera, or `q` to quit. (The keys
fail to work occasionally (frequently :weary:), we are looking for a fix (perhaps multithreading).)

## Credits

We used a lot of code from **[facenet-darknet-inference](https://github.com/lincolnhard/facenet-darknet-inference)**
and **[PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)**. To be more specific, we used
the Facenet config file (`facenet.cfg`) from **facenet-darknet-inference** and used their `test.cpp`
and `face_io.c` as a reference for implementing our `main.py` and `face.py`. We used most darknet code from
**PyTorch-YOLOv3**, with slight modifications to fit the config file.


Below are the README files copied from these two original repos. Thank you!

# PyTorch-YOLOv3
Minimal implementation of YOLOv3 in PyTorch.

## Table of Contents
- [PyTorch-YOLOv3](#pytorch-yolov3)
  * [Table of Contents](#table-of-contents)
  * [Paper](#paper)
  * [Installation](#installation)
  * [Inference](#inference)
  * [Test](#test)
  * [Train](#train)
  * [Credit](#credit)

## Paper
### YOLOv3: An Incremental Improvement
_Joseph Redmon, Ali Farhadi_ <br>

**Abstract** <br>
We present some updates to YOLO! We made a bunch
of little design changes to make it better. We also trained
this new network that’s pretty swell. It’s a little bigger than
last time but more accurate. It’s still fast though, don’t
worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP,
as accurate as SSD but three times faster. When we look
at the old .5 IOU mAP detection metric YOLOv3 is quite
good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared
to 57.5 AP50 in 198 ms by RetinaNet, similar performance
but 3.8× faster. As always, all the code is online at
https://pjreddie.com/yolo/.

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Original Implementation]](https://github.com/pjreddie/darknet)

## Installation
    $ git clone https://github.com/eriklindernoren/PyTorch-YOLOv3
    $ cd PyTorch-YOLOv3/
    $ sudo pip3 install -r requirements.txt

##### Download pretrained weights
    $ cd weights/
    $ bash download_weights.sh

##### Download COCO
    $ cd data/
    $ bash get_coco_dataset.sh

## Inference
Uses pretrained weights to make predictions on images. Below table displays the inference times when using as inputs images scaled to 256x256. The ResNet backbone measurements are taken from the YOLOv3 paper. The Darknet-53 measurement marked shows the inference time of this implementation on my 1080ti card.

| Backbone                | GPU      | FPS      |
| ----------------------- |:--------:|:--------:|
| ResNet-101              | Titan X  | 53       |
| ResNet-152              | Titan X  | 37       |
| Darknet-53 (paper)      | Titan X  | 76       |
| Darknet-53 (this impl.) | 1080ti   | 74       |

    $ python3 detect.py --image_folder /data/samples

<p align="center"><img src="assets/giraffe.png" width="480"\></p>
<p align="center"><img src="assets/dog.png" width="480"\></p>
<p align="center"><img src="assets/traffic.png" width="480"\></p>
<p align="center"><img src="assets/messi.png" width="480"\></p>

## Test
Evaluates the model on COCO test.

    $ python3 test.py --weights_path weights/yolov3.weights

| Model                   | mAP (min. 50 IoU) |
| ----------------------- |:----------------:|
| YOLOv3 (paper)          | 57.9             |
| YOLOv3 (this impl.)     | 58.2             |

## Train
Data augmentation as well as additional training tricks remains to be implemented. PRs are welcomed!
```
    train.py [-h] [--epochs EPOCHS] [--image_folder IMAGE_FOLDER]
                [--batch_size BATCH_SIZE]
                [--model_config_path MODEL_CONFIG_PATH]
                [--data_config_path DATA_CONFIG_PATH]
                [--weights_path WEIGHTS_PATH] [--class_path CLASS_PATH]
                [--conf_thres CONF_THRES] [--nms_thres NMS_THRES]
                [--n_cpu N_CPU] [--img_size IMG_SIZE]
                [--checkpoint_interval CHECKPOINT_INTERVAL]
                [--checkpoint_dir CHECKPOINT_DIR]
```

## Credit
```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```

# facenet-darknet-inference
Face recognition using facenet

**1. Intro**

[Facenet](https://github.com/davidsandberg/facenet) is developed by Google in 2015, the result of the net is the Euclidean embedding of human face. 

By careful defined triplet loss function, facenet achieves high accuracy on LFW(0.9963) and FacesDB(0.9512).

[Darknet](https://github.com/pjreddie/darknet) is a fast, easy to read DL framework. [Yolo](https://pjreddie.com/darknet/yolo/) is running based on it.

**2. Dependencies**

[OpenCV](https://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html) for video i/o, face detection, image resizing, warping, and 3D pose estimation.

[Dlib](https://github.com/davisking/dlib) for facial landmark detection.

[NNPACK](https://github.com/digitalbrain79/NNPACK-darknet) for faster neural network computations.

Zenity for text input.

**3. Installation and run**
```
sudo apt-get install zenity
cd facenet-darknet-inference
#edit makefile
#specify your OPENCV_HEADER_DIR, OPENCV_LIBRARY_DIR, DLIB_HEADER_DIR, DLIB_LIBRARY_DIR, NNPACK_HEADER_DIR, NNPACK_LIBRARY_DIR
make
mkdir data
cd data
touch name
cd ..
mkdir model
```

download [weights](https://drive.google.com/open?id=1ATzb5ZEQo424wlSY-cdlT54FUWlIry8V) and extract in facenet-darknet-inference folder

```
cd facenet-darknet-inference
./facenet-darknet-inference
```

**4. Note**

OpenCV VJ face + Dlib landmark detection is used rather than MTCNN. VJ method is faster, but the unstable cropping may slightly influence recognition accuracy.

KNN is the final classification method, but it is suffered for openset problem. The 1792-d feature before bottleneck layer with normalization is used for KNN, because it has better result in openset than original facenet model, but you can still try the original network configure yourself just replacing *facenet.cfg* to *facenet_full.cfg*

The *facenet.weight* is converted from [facenet inception-resnet v1 20180402-114759 model](https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view)

**5. Result**

![peek 2018-04-19 14-11](https://user-images.githubusercontent.com/16308037/38980107-89460dd4-43ee-11e8-997d-5ceafd226f43.gif)
