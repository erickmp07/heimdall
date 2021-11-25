# heimdall

A Tensorflow application to detect person alive by the camera using a Siamese Convolutional Net and LFW Dataset.

The heimdall is an application developed in Python version 3.7.9 64-bit.

## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [Technologies](#technologies)
- [Contributing](#contributing)
- [License](#license)

## Install

Prerequisites:

Download and install Python [3.7.9](https://www.python.org/downloads/release/python-379/), CUDA Toolkit [11.4](https://developer.nvidia.com/cuda-11-4-3-download-archive) and cuDNN SDK [8.2.2](https://developer.nvidia.com/rdp/cudnn-archive#a-collapse822-114).

NOTE: To run this application, it's recommended to have a GPU microprocessor and you should have a cam.

- First, clone the repository:
```bash
git clone https://github.com/erickmp07/heimdall.git
cd heimdall
```

### YOLO Proof Of Concept

- Download the official weights pretrained on COCO Dataset [here](https://pjreddie.com/media/files/yolov3.weights).

- Save the file with the weights downloaded in the [weights folder](YOLOv3/weights).

Then:

- Install the dependencies with `pip command`:
```bash
pip install -r requirements.txt
```

### Face ID project

- Download the LFW Dataset [here](http://vis-www.cs.umass.edu/lfw/lfw.tgz).

- Save the TGZ file in the heimdall root path.

- To skip the model training, you can download a model pretrained [here](https://drive.google.com/file/d/1wwLGkIFNFU3osdMRuvjJ23wVE6XsHnZs/view?usp=sharing).

- Extract the model pretrained to the heimdall root path.

Then:

- Install the dependencies with `pip command`:
```bash
pip install -r requirements.txt
```

## Usage

### YOLO Proof Of Concept

- Load the pretrained weights and converts them to the Tensorflow format using the `convert_weights.py` script:
```bash
python convert_weights.py
```

- The pattern command to run the `video.py` script is:
```bash
python video.py <iou threshold> <confidence threshold> <filename>
```

NOTE: The [`video.py`](video.py) script has a GPU check (recommended). So, if you don't have a GPU installed and want to run the model, comment this line at the script:
```python
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
```

- Runs the model using one of following commands, for example:
```bash
python video.py 0.7 0.8 YOLOv3/data/videos/video01.mp4 # For video file input
```

or

```bash
python video.py 0.8 0.9 0 # For cam input
```

NOTE: Hit the "q" key to quit the detection.

Then, you can find the detections in the [videos folder](YOLOv3/data/videos) where the file name has the sufix "_detection" and the extension ".avi".

NOTE: Only one video can be processed at one run.

### Face ID project

- The pattern command to run the `face_id.py` script is:
```bash
python face_id.py <detection threshold> <verification threshold> <train flag> <test flag>
```

All the arguments are "optional" and the default values are:
```python
detection_threshold = 0.5
verification_threshold = 0.5
train_flag = ''
test_flag = ''
```

The command below run the [`face_id.py`](face_id.py) script with the default values and DON'T train the model and DON'T make predictions with the test data:
```bash
python face_id.py
```

The command below run the [`face_id.py`](face_id.py) script with values to detection and verification thresholds, DON'T train the model and make predictions with the test data:
```bash
python face_id.py 0.8 0.7 '' test 
```

The command below run the [`face_id.py`](face_id.py) script with values to detection and verification thresholds, train the model and make predictions with the test data:
```bash
python face_id.py 0.8 0.7 train test 
```

After run one of the flavors of the `face_id` command, a dialog will open with your cam image. 
You should collect some positive (hitting the "p" key) and anchor (hitting the "a" key) examples. To quit the image collector, hit the "q" key.

Then, a new dialog will open with your cam image. Now, you can test the facial recognition.
Hit the "v" key for verify if your face will be recognized or hit the "q" key for quit the recognition.

## Technologies

This project was developed with the following technologies:

- [Python](https://www.python.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [NumPy](https://numpy.org/)
- [opencv-python](https://github.com/opencv/opencv-python)
- [Matplotlib](https://matplotlib.org/)

## Contributing

NOTE: This is a graduation project from the Computer Engineering course at PUC-Rio.

To ask a question, please [contact me](mailto:erimacedo_92@hotmail.com).

## License

Licensed under [MIT](LICENSE) license.
