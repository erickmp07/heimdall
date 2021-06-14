"""Contains utility functions for Yolo v3 model."""

import tensorflow as tf
import numpy as np
import cv2

def non_max_suppression(
    inputs, 
    model_size, 
    max_output_size, 
    max_output_size_per_class, 
    iou_threshold, 
    confidence_threshold):
    """Clears unnecessary bounding boxes detections.

    Args:
        inputs: The model's prediction.
        model_size: The size of the model.
        max_output_size_per_class: Max output size per class.
        iou_threshold: Intersection Over Union (IOU) threshold.
        confidence_threshold: Confidence threshold.
    
    Returns:
        Multiple values:
            - boxes: Boxes selected by the algorithm.
            - scores: Scores of each box.
            - classes: Classes found in the prediction.
            - valid_detections: Valid detections.
    """

    # Creates and normalizes the bounding boxes
    bounding_boxes, confidences, class_probability = tf.split(inputs, [4, 1, -1], axis=-1)
    bounding_boxes = bounding_boxes / model_size[0]

    # Calcultes the scores
    scores = confidences * class_probability

    # Calls the Tensorflow non_max_suppression() method to calculate the values for us
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bounding_boxes, (tf.shape(bounding_boxes)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=max_output_size_per_class,
        max_total_size=max_output_size,
        iou_threshold=iou_threshold,
        score_threshold=confidence_threshold)

    return boxes, scores, classes, valid_detections

def resize_image(inputs, model_size):
    """Resizes the image to fit with the model's size.

    Args:
        inputs: The image read.
        model_size: The size of the model.

    Returns:
        The image resized for the model size.
    """

    inputs = tf.image.resize(inputs, model_size)

    return inputs

def load_class_names(file_name):
    """Loads the classes names used in object detection.
    
    Args:
        file_name: The name of the file that lists de classes used in the detection.
    
    Returns:
        The names of the classes.
    """

    with open(file_name, 'r') as file:
        class_names = file.read().splitlines()

    return class_names

def output_boxes(
    inputs,
    model_size, 
    max_output_size,
    max_output_size_per_class,
    iou_threshold,
    confidence_threshold):
    """Converts the boxes into the format of (top-left-corner, bottom-right-corner).
    
    Args:
        inputs: The model's prediction.
        model_size: The size of the model.
        max_output_size_per_class: Max output size per class.
        iou_threshold: Intersection Over Union (IOU) threshold.
        confidence_threshold: Confidence threshold.
    
    Returns:
        Multiple values:
            - boxes: Boxes selected by the algorithm.
            - scores: Scores of each box.
            - classes: Classes found in the prediction.
            - valid_detections: Valid detections.
    """

    # Splits the model's prediction and identifies the class position
    center_x, center_y, width, height, confidence, classes = tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)

    # Creates the new format for the box
    top_left_x = center_x - width / 2.0
    top_left_y = center_y - height / 2.0
    bottom_right_x = center_x + width / 2.0
    bottom_right_y = center_y + height / 2.0

    inputs = tf.concat([top_left_x, top_left_y, bottom_right_x, bottom_right_y, confidence, classes], axis=-1)

    # Creates the boxes of the prediction done
    boxes_dicts = non_max_suppression(
        inputs=inputs,
        model_size=model_size,
        max_output_size=max_output_size,
        max_output_size_per_class=max_output_size_per_class,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold)

    return boxes_dicts

def draw_outputs(image, boxes, objectness, classes, numbers, class_names):
    """Draws boxes, classes and scores on an image.

    Args:
        image: The image (frame) where some information will be drawn.
        boxes: Boxes where the classes were identified.
        objectness: The scores calculated on the prediction.
        classes: The classes identified.
        numbers: Valid detections calculated by the non_max_suppression algorithm.
        class_names: The names of the classes identified.

    Returns:
        The image (frame) drawn. 
    """

    boxes, objectness, classes, numbers = boxes[0], objectness[0], classes[0], numbers[0]

    boxes = np.array(boxes)

    for i in range(numbers):
        x1y1 = tuple((boxes[i, 0:2] * [image.shape[1], image.shape[0]]).astype(np.int32))
        x2y2 = tuple((boxes[i, 2:4] * [image.shape[1], image.shape[0]]).astype(np.int32))

        image = cv2.rectangle(image, (x1y1), (x2y2), (255, 0, 0), 2)

        class_names_index = int(classes[i])

        if (len(class_names) <= class_names_index):
            break

        image = cv2.putText(image, '{}: {:.2f}%'.format(
            class_names[class_names_index],
            objectness[i] * 100),
                (x1y1),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1,
                (0, 0, 255),
                2)
                
    return image