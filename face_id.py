import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import tarfile
import uuid
from datetime import datetime
import sys

from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.python.data.ops.dataset_ops import PrefetchDataset, ShuffleDataset
from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.python.training.tracking.util import Checkpoint
import tensorflow as tf


# Create folders structures
POSITIVE_PATH = os.path.join('data', 'positive')
NEGATIVE_PATH = os.path.join('data', 'negative')
ANCHOR_PATH = os.path.join('data', 'anchor')
VERIFICATION_PATH = os.path.join('application_data', 'verification_images')
INPUT_PATH = os.path.join('application_data', 'input_image')

# Train constants
EPOCHS = 50

MODEL_NAME = 'siamesemodel.h5'



class L1Dist(Layer):
    """
    Siamese L1 Distance class
    """

    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        """
        Similarity calculation
        """
        return tf.math.abs(input_embedding - validation_embedding)


def set_gpu_growth():
    """ 
    Avoid Out Of Memory errors by setting GPU Memory Consumption Growth
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def make_data_dirs():
    # Make the directories
    if (not os.path.exists(POSITIVE_PATH)):
        print(f'{datetime.now()} : INFO - Making POSITIVE directory > {POSITIVE_PATH}')
        os.makedirs(POSITIVE_PATH)

    if (not os.path.exists(NEGATIVE_PATH)):
        print(f'{datetime.now()} : INFO - Making NEGATIVE directory > {NEGATIVE_PATH}')
        os.makedirs(NEGATIVE_PATH)

    if (not os.path.exists(ANCHOR_PATH)):
        print(f'{datetime.now()} : INFO - Making ANCHOR directory > {ANCHOR_PATH}')
        os.makedirs(ANCHOR_PATH)

    if (not os.path.exists(VERIFICATION_PATH)):
        print(f'{datetime.now()} : INFO - Making VERIFICATION directory > {VERIFICATION_PATH}')
        os.makedirs(VERIFICATION_PATH)

    if (not os.path.exists(INPUT_PATH)):
        print(f'{datetime.now()} : INFO - Making INPUT directory > {INPUT_PATH}')
        os.makedirs(INPUT_PATH)


def extract_lfw_data():
    """ 
    Uncompress TGZ file with the Labelled Faces Wild Dataset
    """
    if (not os.path.exists('lfw')):
        print(f"{datetime.now()} : INFO - Extracting the LFW Dataset files to the lfw directory. It's may take a while...")

        tar = tarfile.open("lfw.tgz", "r:gz")
        tar.extractall()
        tar.close()

        print(f'{datetime.now()} : INFO - Extraction completed.')


def move_lfw_images_to_negative_folder():
    """ 
    Move LFW Images to the following repository data/negative
    """
    directories_count = len(os.listdir('lfw'))

    current_dir_index = 1

    for directory in os.listdir('lfw'):
        print(f'{datetime.now()} : INFO - Moving LFW files to NEGATIVE directory > Directory {current_dir_index} of {directories_count}')

        for file in os.listdir(os.path.join('lfw', directory)):
            OLD_PATH = os.path.join('lfw', directory, file)
            NEW_PATH = os.path.join(NEGATIVE_PATH, file)
            os.replace(OLD_PATH, NEW_PATH)

        current_dir_index += 1


def data_augmentation(image):
    data = []
    for i in range(9):
        image = tf.image.stateless_random_brightness(image, max_delta=0.02, seed=(1, 2))
        image = tf.image.stateless_random_contrast(image, lower=0.6, upper=1, seed=(1, 3))
        image = tf.image.stateless_random_flip_left_right(image, seed=(np.random.randint(100), np.random.randint(100)))
        image = tf.image.stateless_random_jpeg_quality(image, min_jpeg_quality=90, max_jpeg_quality=100, seed=(np.random.randint(100), np.random.randint(100)))
        image = tf.image.stateless_random_saturation(image, lower=0.9, upper=1, seed=(np.random.randint(100), np.random.randint(100)))

        data.append(image)
        
    return data


def verify(model, detection_threshold, verification_threshold):
    # Build results array
    results = []

    for image in os.listdir(VERIFICATION_PATH):
        input_image = preprocess(os.path.join(INPUT_PATH, 'input_image.jpg'))
        validation_image = preprocess(os.path.join(VERIFICATION_PATH, image))

        # Make predictions
        result = model.predict(list(np.expand_dims([input_image, validation_image], axis=1)))

        results.append(result)

    # Detection Threshold: Metric above which a prediction is considered positive
    detection = np.sum(np.array(results) > detection_threshold)
    
    # Verification Threshold: Proportion of positive predictions / total positive samples
    verification = detection / len(os.listdir(VERIFICATION_PATH))
    verified = verification > verification_threshold

    return results, verified


def connect_to_cam_to_collect_images():
    """ 
    Establish a connection to the webcam
    """
    print(f'{datetime.now()} : INFO - Opening the cam')
    print(f"{datetime.now()} : INFO - Type 'a' for collect anchor image; 'p' for collect positive image; 'q' for quit")

    capture = cv2.VideoCapture(0)
    while capture.isOpened():
        ret, frame = capture.read()

        # Cut down frame to 250x250px
        frame = frame[120:120+250, 200:200+250, :]

        # Collect anchors
        if cv2.waitKey(1) & 0xFF == ord('a'):
            # Create the unique file path
            imgname = os.path.join(ANCHOR_PATH, '{}.jpg'.format(uuid.uuid4()))

            # Write out anchor image
            cv2.imwrite(imgname, frame)

        # Collect positives
        if cv2.waitKey(1) & 0xFF == ord('p'):
            # Create the unique file path
            imgname = os.path.join(
                POSITIVE_PATH, '{}.jpg'.format(uuid.uuid4()))

            # Write out anchor image
            cv2.imwrite(imgname, frame)

        # Show image back to screen
        cv2.imshow('Image Collection', frame)

        # Breaking gracefully
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam
    capture.release()

    # Close the image show frame
    cv2.destroyAllWindows()


def connect_to_cam_to_verify(siamese_model, detection_threshold, verification_threshold):
    """ 
    Establish a connection to the webcam
    """
    print(f'{datetime.now()} : INFO - Opening the cam')
    print(f"{datetime.now()} : INFO - Type 'v' for verify; 'q' for quit")

    capture = cv2.VideoCapture(0)
    while capture.isOpened():
        ret, frame = capture.read()

        # Cut down frame to 250x250px
        frame = frame[120:120+250, 200:200+250, :]

        # Verification trigger
        if cv2.waitKey(1) & 0xFF == ord('v'):
            # Save input image to application_data/input_image folder
            cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)

            print(f"{datetime.now()} : Doing Verification...")

            # Run verification
            results, verified = verify(siamese_model, detection_threshold, verification_threshold)

            print(f"{datetime.now()} : Verification result: {verified}")
            print(f"{datetime.now()} : INFO - Type 'v' for verify; 'q' for quit")


        # Show image back to screen
        cv2.imshow('Verification', frame)

        # Breaking gracefully
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam
    capture.release()

    # Close the image show frame
    cv2.destroyAllWindows()


def augmentation_images_collected(path: str):
    print(f"{datetime.now()} : INFO - Doing image augmentation from path '{path}'. It's may take a while...")

    augmented_images = []
    
    for file_name in os.listdir(os.path.join(path)):
        image_path = os.path.join(path, file_name)
        image = cv2.imread(image_path)
        augmented_images.append(data_augmentation(image))

    for images in augmented_images:
        for image in images:
            cv2.imwrite(os.path.join(path, '{}.jpg'.format(uuid.uuid4())), np.array(image))


def preprocess(file_path):
    """
    Scale and rezie an image to 0..1 range
    """

    # Read in image from file path
    byte_image = tf.io.read_file(file_path)

    # Load in the image
    image = tf.io.decode_jpeg(byte_image)

    # Preprocessing steps - resizing the image to be 100x100x3
    image = tf.image.resize(image, (100, 100))

    # Scale image to be between 0 and 1
    image = image / 255.0

    return image


def preprocess_twin(input_image, validation_image, label):
    return (preprocess(input_image), preprocess(validation_image), label)


def build_dataloader():
    """
    Build dataloader pipeline
    """
    anchor_length = len(os.listdir(ANCHOR_PATH))
    positive_length = len(os.listdir(POSITIVE_PATH))
    negative_length = len(os.listdir(NEGATIVE_PATH))

    min_path_length = min(anchor_length, positive_length, negative_length)
    number_of_images_to_take = round(min_path_length * 0.85)

    anchor = tf.data.Dataset.list_files(ANCHOR_PATH+'\*.jpg').take(number_of_images_to_take)
    positive = tf.data.Dataset.list_files(POSITIVE_PATH+'\*.jpg').take(number_of_images_to_take)
    negative = tf.data.Dataset.list_files(NEGATIVE_PATH+'\*.jpg').take(number_of_images_to_take)

    # Take some positive examples and copy to verification folder
    images_to_copy_length = len(os.listdir(POSITIVE_PATH)[:round(min_path_length * 0.015)])

    for index, image_name in enumerate(os.listdir(POSITIVE_PATH)[:round(min_path_length * 0.015)]):
        image_path = os.path.join(POSITIVE_PATH, image_name)
        image = cv2.imread(image_path)
        cv2.imwrite(os.path.join(VERIFICATION_PATH, image_name), np.array(image))
        print(f"{datetime.now()} : INFO - Copying images from POSITIVE path to VERIFICATION path: {index + 1} of {images_to_copy_length} copied.")

    positives = tf.data.Dataset.zip(
        (anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))

    negatives = tf.data.Dataset.zip(
        (anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))

    data = positives.concatenate(negatives)

    # Build dataloader pipeline
    data = data.map(preprocess_twin)
    data = data.cache()
    data = data.shuffle(buffer_size=10000)

    return data


def build_training_partition(data: ShuffleDataset):
    """
    Build the training partition
    """
    train_data = data.take(round(len(data) * 0.7))
    train_data = train_data.batch(16)
    train_data = train_data.prefetch(8)

    return train_data


def build_testing_partition(data: ShuffleDataset):
    """
    Build the testing partition
    """
    test_data = data.skip(round(len(data) * 0.7))
    test_data = test_data.take(round(len(data) * 0.3))
    test_data = test_data.batch(16)
    test_data = test_data.prefetch(8)

    return test_data


def make_embedding():
    input_layer = Input(shape=(100, 100, 3), name='input_image')

    # First block
    c1 = Conv2D(64, (10, 10), activation='relu')(input_layer)
    m1 = MaxPooling2D(64, (2, 2), padding='same')(c1)

    # Second block
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2, 2), padding='same')(c2)

    # Third block
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2, 2), padding='same')(c3)

    # Final embedding block
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=[input_layer], outputs=[d1], name='embedding')


def make_siamese_model(embedding: Model):

    # Anchor image input in the network
    input_image = Input(name='input_image', shape=(100, 100, 3))

    # Validation image in the network
    validation_image = Input(name='validation_image', shape=(100, 100, 3))

    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    
    distances = siamese_layer(embedding(input_image), embedding(validation_image))

    # Classification layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


@tf.function
def train_step(batch, siamese_model: Model, binary_cross_loss: BinaryCrossentropy, optimizer: Adam):

    # Record allo of our operations
    with tf.GradientTape() as tape:
        # Get anchor and positive/negative image
        x = batch[:2]

        # Get label
        y = batch[2]

        # Forward pass
        yhat = siamese_model(x, training=True)

        # Calculate Loss
        loss = binary_cross_loss(y, yhat)

    print(f'Loss: {loss}')

    # Calculate gradients
    gradient = tape.gradient(loss, siamese_model.trainable_variables)

    # Calculate updated weights and apply to siamese model 
    optimizer.apply_gradients(zip(gradient, siamese_model.trainable_variables))

    return loss


def train(data: PrefetchDataset, EPOCHS: int, siamese_model: Model, binary_cross_loss: BinaryCrossentropy, 
    optimizer: Adam, checkpoint: Checkpoint, checkpoint_prefix: str):
    # Loop through epochs
    for epoch in range(1, EPOCHS + 1):
        print(f'\n Epoch {epoch} of {EPOCHS}')
        
        progress_bar = tf.keras.utils.Progbar(len(data))

        # Loop through each batch
        for index, batch in enumerate(data):
            # Run train step here
            train_step(batch, siamese_model, binary_cross_loss, optimizer)
            progress_bar.update(index + 1)

        # Save checkpoints
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


def make_prediction_to_test(test_data, siamese_model):
    print(f"{datetime.now()} : INFO - Making predictions with the test data")
    # Get a batch of test data
    test_input, test_val, y_true = test_data.as_numpy_iterator().next()

    # Make predictions
    y_hat = siamese_model.predict([test_input, test_val])

    # Post processing the results
    [1 if prediction > 0.5 else 0 for prediction in y_hat]

    # Creating a metric object
    recall = Recall()

    # Calculating the recall value
    recall.update_state(y_true, y_hat)

    # Return Recall result
    recall_result = recall.result().numpy()

    print(f'Recall result: {recall_result}')

    # Creating a metric object
    precision = Precision()

    # Calculating the recall value
    precision.update_state(y_true, y_hat)

    # Return Recall result
    precision_result = precision.result().numpy()

    print(f'Precision result: {precision_result}')

    for index, predict_result in enumerate(y_true):
        if (predict_result == 1):
            print('Plotting twin images')
        else:
            print('Plotting different images')

        # Set plot size
        plt.figure(figsize=(10, 8))

        # Set first subplot
        plt.subplot(1, 2, 1)
        plt.imshow(test_input[index])

        # Set second subplot
        plt.subplot(1, 2, 2)
        plt.imshow(test_val[index])

        # Renders
        plt.show()


def main(detection_threshold, verification_threshold, train_flag, test_flag):
    set_gpu_growth()

    make_data_dirs()

    extract_lfw_data()

    move_lfw_images_to_negative_folder()

    connect_to_cam_to_collect_images()

    #augmentation_images_collected(ANCHOR_PATH)
    #augmentation_images_collected(POSITIVE_PATH)

    data = build_dataloader()

    train_data = build_training_partition(data)

    test_data = build_testing_partition(data)

    embedding = make_embedding()

    siamese_model = make_siamese_model(embedding)

    binary_cross_loss = tf.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint_directory = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_directory, 'ckpt')
    checkpoint = tf.train.Checkpoint(opt=optimizer, siamese_model=siamese_model)

    if (train_flag == 'train'):
        train(train_data, EPOCHS, siamese_model, binary_cross_loss, optimizer, checkpoint, checkpoint_prefix)

    # Load the the model or the latest checkpoint
    if (os.path.exists(MODEL_NAME)):
        siamese_model = tf.keras.models.load_model(MODEL_NAME, custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})
    else:
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory)).expect_partial()
        
        siamese_model = checkpoint.siamese_model
        optimizer = checkpoint.opt

    if (test_flag == 'test'):
        make_prediction_to_test(test_data, siamese_model)

    # Save weights
    print(f"{datetime.now()} : INFO - Saving the model with name: '{MODEL_NAME}'")
    siamese_model.save(MODEL_NAME)
    print(f"{datetime.now()} : INFO - Model '{MODEL_NAME}' saved")

    print(f"{datetime.now()} : INFO - Opening the verification mode")

    connect_to_cam_to_verify(siamese_model, detection_threshold, verification_threshold)


if __name__ == '__main__':
    argv_length = len(sys.argv)

    if argv_length == 1:
        main(0.5, 0.5, '', '')
    elif argv_length == 2:
        main(float(sys.argv[1]), 0.5, '', '')
    elif argv_length == 3:
        main(float(sys.argv[1]), float(sys.argv[2]), '', '')
    elif argv_length == 4:
        main(float(sys.argv[1]), float(sys.argv[2]), sys.argv[3], '')
    elif argv_length == 5:
        main(float(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4])
