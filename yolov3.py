"""Contains YOLOv3 core definitions."""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Input, ZeroPadding2D, LeakyReLU, UpSampling2D

def parse_cfg(cfgfile):
    """ Parses the configuration file.

    Args:
        cfgfile: The YOLOv3 configuration file.

    Returns:
        The list of blocks of the configuration file parsed.
    """

    # Opens the configuration file and read it removing unnecessary characters
    # like '\n' and '#'
    with open(cfgfile, 'r') as file:
        # Stores all the lines of the configuration file
        lines = [line.rstrip('\n') for line in file if line != '\n' and line[0] != '#']
    
    holder = {}
    blocks = []
    
    # Loops over lines reading every attribute and stores them in the blocks
    for line in lines:
        if line[0] == '[':
            line = 'type=' + line[1:-1].rstrip()
            
            if len(holder) != 0:
                blocks.append(holder)
                holder = {}
        
        key, value = line.split("=")
        holder[key.rstrip()] = value.lstrip()
    
    blocks.append(holder)

    return blocks


def YOLOv3Net(cfgfile, model_size, number_of_classes):
    """ Builds the YOLOv3 Network.

    Args:
        cfgfile: The YOLOv3 configuration file.
        model_size: The size of the Model to be created.
        number_of_classes: The number of the classes that the Model should have.
    
    Returns:
        The YOLOv3 Network Model created.
    """

    blocks = parse_cfg(cfgfile)

    outputs = {}
    output_filters = []
    filters = []
    out_prediction = []
    scale = 0

    # Defines the input model using Keras function
    inputs = input_image = Input(shape=model_size)
    # Normalizes it to the range of 0-1
    inputs = inputs / 255.0

    # Iterates over blocks checking the type of the block wich corresponds to the type of the layer
    for i, block in enumerate(blocks[1:]):
        # If it's a convolutional layer
        if (block["type"] == "convolutional"):
            # In the convolutional block, it'll find the following attributes:
            # - batch_normalize;
            # - activation;
            # - filters;
            # - pad;
            # - size;
            # - stride

            activation = block["activation"]
            filters = int(block["filters"])
            kernel_size = int(block["size"])
            strides = int(block["stride"])

            # Checks whether the stride is greater than 1. If it's true, then downsampling is performed, so adjusts the padding
            if strides > 1:
                inputs = ZeroPadding2D(((1, 0), (1, 0)))(inputs)

            inputs = Conv2D(filters,
                            kernel_size,
                            strides=strides,
                            padding='valid' if strides > 1 else 'same',
                            name='conv_' + str(i),
                            use_bias=False if ("batch_normalize" in block) else True)(inputs)

            # If bactch_normalize is find in a block, then add layers BatchNormalization and LeakyReLU
            if "batch_normalize" in block:
                inputs = BatchNormalization(name='bnorm_' + str(i))(inputs)
            if activation == "leaky":
                inputs = LeakyReLU(alpha=0.1, name='leaky_' + str(i))(inputs)

        # If it's a upsample layer
        elif (block["type"] == "upsample"):
            stride = int(block["stride"])

            # Adds layer UpSampling2D
            inputs = UpSampling2D(stride)(inputs)

        # If it's a route layer
        elif (block["type"] == "route"):
            block["layers"] = block["layers"].split(',')
            start = int(block["layers"][0])

            # Layer with one value means that if we are in this route block, we need to backward 'start' layers and 
            # then output the feature map from that layer
            if len(block["layers"]) > 1:
                end = int(block["layers"][1]) - i
                filters = output_filters[i + start] + output_filters[end]  # Index negatif :end - index
                inputs = tf.concat([outputs[i + start], outputs[i + end]], axis=-1)
            # If layers has two values, we need to concatenate the feature map from a previous layer and the 
            # feature map from layer the followig layer
            else:
                filters = output_filters[i + start]
                inputs = outputs[i + start]

        # If it's a shortcut layer
        elif block["type"] == "shortcut":
            # Performs a skip connection
            # Backward from_ layers, then take the feature map from that layer, and add it with the feature map
            # from the previous layer
            from_ = int(block["from"])
            inputs = outputs[i - 1] + outputs[i + from_]

        # If it's yolo detection layer
        elif block["type"] == "yolo":
            # Takes all the necessary attributes associated with it
            mask = block["mask"].split(",")
            mask = [int(x) for x in mask]
            anchors = block["anchors"].split(",")
            anchors = [int(anchor) for anchor in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            number_of_anchors = len(anchors)

            # Reshapes the YOLOv3 output to the form of [None, B * grid size * grid size, 5 * C],
            # where B is the number of anchors and C is the number of classes
            out_shape = inputs.get_shape().as_list()

            inputs = tf.reshape(inputs, [-1, number_of_anchors * out_shape[1] * out_shape[2], \
										 5 + number_of_classes])

            # Access all boxes attributes
            box_centers = inputs[:, :, 0:2]
            box_shapes = inputs[:, :, 2:4]
            confidence = inputs[:, :, 4:5]
            classes = inputs[:, :, 5:number_of_classes + 5]

            # Refines Bounding Boxes
            # Uses the sigmoid function to convert box_centers, confidence and classes value into range of 0-1
            box_centers = tf.sigmoid(box_centers)
            confidence = tf.sigmoid(confidence)
            classes = tf.sigmoid(classes)

            # Converts box_shapes
            anchors = tf.tile(anchors, [out_shape[1] * out_shape[2], 1])
            box_shapes = tf.exp(box_shapes) * tf.cast(anchors, dtype=tf.float32)

            # Uses a meshgrid to convert the relative positions of the center boxes into the real positions
            x = tf.range(out_shape[1], dtype=tf.float32)
            y = tf.range(out_shape[2], dtype=tf.float32)

            cx, cy = tf.meshgrid(x, y)
            cx = tf.reshape(cx, (-1, 1))
            cy = tf.reshape(cy, (-1, 1))
            cxy = tf.concat([cx, cy], axis=-1)
            cxy = tf.tile(cxy, [1, number_of_anchors])
            cxy = tf.reshape(cxy, [1, -1, 2])

            strides = (input_image.shape[1] // out_shape[1], \
                       input_image.shape[2] // out_shape[2])
            box_centers = (box_centers + cxy) * strides

            # Then, concatenate them all together
            prediction = tf.concat([box_centers, box_shapes, confidence, classes], axis=-1)

            # YOLOv3 does 3 predictions across the scale
            if scale:
                out_prediction = tf.concat([out_prediction, prediction], axis=1)
            else:
                out_prediction = prediction
                scale = 1

        # Since the route and shortcut layers need output feature maps from previous layers, so for every
        # iteration, we always keep the track of the feature maps and output filters
        outputs[i] = inputs
        output_filters.append(filters)

    # Returns the Model
    model = Model(input_image, out_prediction)
    model.summary()

    return model
