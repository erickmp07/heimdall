"""Loads Yolo v3 pretrained weights and saves them in tensorflow format."""

import numpy as np
from yolov3 import YOLOv3Net
from yolov3 import parse_cfg

def load_weights(model, cfgfile, weightfile):
    """Reshapes and loads official pretrained Yolo weights.
    
    Args:
        model: Parameters of the network's model.
        cfgfile: The YOLOv3 configuration file.
        weightfile: The YOLOv3 official weight file.
    
    Returns:
        None.
    """

    # Open the weights file
    fp = open(weightfile, "rb")

    # Reads the first 5 values (header information) - Skipped
    np.fromfile(fp, dtype=np.int32, count=5)

    # Reads the rest of the values - the weights
    blocks = parse_cfg(cfgfile)

    # Loops over the blocks and searchs for the convolutional layer
    for i, block in enumerate(blocks[1:]):
        # If it's a convolutional layer
        if (block["type"] == "convolutional"):
            convolutional_layer = model.get_layer('conv_' + str(i))
            print("layer: ", i + 1, convolutional_layer)

            filters = convolutional_layer.filters
            kernel_size = convolutional_layer.kernel_size[0]
            input_shape = convolutional_layer.input_shape[-1]

            # If bactch_normalize is find in a block, reads the weights as [gamma, beta, mean, variance].
            # Otherwise, reads the weights as the bias
            if "batch_normalize" in block:
                normalize_layer = model.get_layer('bnorm_' + str(i))
                print("layer: ", i + 1, normalize_layer)

                # Reads the batch_normalize weights
                batch_normalize_weights = np.fromfile(fp, dtype=np.float32, count=4 * filters)

                # tf [gamma, beta, mean, variance]
                batch_normalize_weights = batch_normalize_weights.reshape((4, filters))[[1, 0, 2, 3]]
            else:
                convolutional_bias = np.fromfile(fp, dtype=np.float32, count=filters)
            
            # darknet shape (output_dimension, input_dimension, height, width)
            convolutional_shape = (filters, input_shape, kernel_size, kernel_size)
            
            # Reads the convolutional weights
            convolutional_weights = np.fromfile(fp, dtype=np.float32, count=np.product(convolutional_shape))

            # tensorflow shape (height, width, input_dimension, output_dimension)
            convolutional_weights = convolutional_weights.reshape(convolutional_shape).transpose([2, 3, 1, 0])

            # If batch_normalize is find in a block, sets [beta, gamma, mean, variance] as the weights.
            # Otherwise, sets the convolutional bias as the weights
            if "batch_normalize" in block:
                normalize_layer.set_weights(batch_normalize_weights)
                convolutional_layer.set_weights([convolutional_weights])
            else:
                convolutional_layer.set_weights([convolutional_weights, convolutional_bias])

    # Failed to read all data
    assert len(fp.read()) == 0 
    
    # Closes the file
    fp.close()

def main():

    # Weight file path
    weightfile = "YOLOv3/weights/yolov3.weights"
    # Configuration file path
    cfgfile = "YOLOv3/configuration/yolov3.cfg"

    model_size = (416, 416, 3)
    number_of_classes = 80

    # Creates the model
    model = YOLOv3Net(cfgfile, model_size, number_of_classes)

    # Loads the weights
    load_weights(model, cfgfile, weightfile)

    try:
        model.save_weights('YOLOv3/weights/yolov3_weights.tf')
        print('\nThe file \'yolov3_weights.tf\' has been saved successfully.')
    except IOError:
        print("Couldn't write the file \'yolov3_weights.tf\'.")

if __name__ == '__main__':
    main()