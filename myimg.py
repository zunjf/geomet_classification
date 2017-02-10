import tensorflow as tf
import sys, numpy

# dimensions of images
height = 32
width = 32

# number of classes
nClass = 2

# function to tell tensorflow how to read a single image from input file
def getImage(filename):
    # convert filenames to a queue for an input pipeline
    filenameQ = tf.train.string_input_producer([filename], num_epochs=None)

    # object to read records
    recordReader = tf.TFRecordReader()

    # read the full set of features for a single example
    key, fullExample = recordReader.read(filenameQ)

    # parse the full example into its' componen features
    features = tf.parse_single_example(
        fullExample,
        features={
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/colorspace': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/format': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/filename': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value='')
        }
    )

    # manipulate the label and image features
    label = features['image/class/label']
    image_buffer = features['image/encoded']

    # decode the jpeg
    with tf.name_scope('decode_jpeg', [image_buffer], None):
        # decode
        image = tf.image.decode_jpeg(image_buffer, channels=3)

        # convert into single precision data type
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # cast image into a single array, where each element corrensponds to the greyscale
    # value of a single pixel
    # the "1-..." part invert the image, so that the background is black
    image = tf.reshape(1-tf.image.rgb_to_grayscale(image), [height*width])

    # re-define label as 'one-hot' vector
    # it will be [0,1] or [1,0]
    label = tf.pack(tf.one_hot(label-1, nClass))

    return label, image
