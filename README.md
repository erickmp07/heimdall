# heimdall

A Tensorflow application to detect person in video file or alive by the camera using the YOLO (You Only Look Once) v3 Net and COCO Dataset.

The heimdall is an application developed in Python version 3.7.9 64-bit.

## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [Technologies](#technologies)
- [Contributing](#contributing)
- [License](#license)

## Install

Prerequisites:

Download and install [Python](https://www.python.org/downloads/).

- First, clone the repository:
```bash
git clone https://github.com/erickmp07/heimdall.git
cd heimdall
```

- Download the official weights pretrained on COCO dataset [here](https://pjreddie.com/media/files/yolov3.weights):

- Save the file with the weights downloaded in the [weights folder](YOLOv3/weights).

- Install the dependencies with `pip command` or calling the [Makefile](Makefile):
```bash
pip install -r requirements.txt
```

## Usage

- Load the pretrained weights and converts them to the Tensorflow format using the `convert_weights.py` script:
```bash
python convert_weights.py
```

- The pattern command to run the model is:
```bash
python video.py <iou threshold> <confidence threshold> <filename>
```

Note: The `video.py` script has a GPU check (recommended). So, if you don't have a GPU installed and want to run the model, comment these lines at the script:
```python
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
```

- Runs the model using one of following commands, for example:
```bash
python video.py 0.7 0.8 YOLOv3/data/videos/video01.mp4 # For video file input
```

or

```bash
python video.py 0.8 0.9 0 # For cam input
```

Note: Press the "q" key to quit the detection.

Then, you can find the detections in the [videos folder](YOLOv3/data/videos) where the file name has the sufix "_detection" and the extension ".avi".

Note: Only one video can be processed at one run.

## Technologies

This project was developed with the following technologies:

- [Python](https://www.python.org/)
- [Tensorflow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [Numpy](https://numpy.org/)
- [opencv-python](https://github.com/opencv/opencv-python)
- [YOLO v3](https://pjreddie.com/darknet/yolo/)
- [COCO Dataset](https://cocodataset.org/)

## Contributing

Note: This is a graduation project from the Computer Engineering course at PUC-Rio.

PRs and stars are always welcome.

To ask a question, please [contact me](mailto:erimacedo_92@hotmail.com).

## License

Licensed under [MIT](LICENSE) license.
