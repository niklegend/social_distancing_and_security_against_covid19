# Adapted from https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py
import os

import pandas as pd
import tensorflow.compat.v1 as tf
from masterthesis.utils import TimeIt
from object_detection.utils import dataset_util, label_map_util


def create_tf_example(example, images_dir, class_text_to_id):
    img_path = os.path.join(images_dir, example.filename)
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_img = fid.read()
    width, height = example.width, example.height

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for annotation in example.annotations:
        # Convert absolute corners to relative corners
        bbox = annotation['bbox']
        xmin, xmax = tuple(x / width for x in bbox[::2])
        ymin, ymax = tuple(y / height for y in bbox[1::2])
        del bbox

        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)

        class_text = annotation['class']
        classes_text.append(class_text.encode('utf8'))
        classes.append(class_text_to_id(class_text))

    filename = example.filename.encode('utf8')
    source_id = img_path.encode('utf8')
    ext = os.path.splitext(example.filename)[1][1:].lower()  # remove initial point
    image_format = ext.encode('utf8')

    return tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(source_id),
        'image/encoded': dataset_util.bytes_feature(encoded_img),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    })) if len(classes) > 0 else None


def create_tf_record(examples, label_map_dict, images_dir, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with TimeIt(f'Successfully created TensorFlow record: {output_path}'):
        with tf.python_io.TFRecordWriter(output_path) as writer:
            images_dir = os.path.join(images_dir)
            for _, example in examples.iterrows():
                tf_example = create_tf_example(
                    example,
                    images_dir,
                    lambda class_text: label_map_dict[class_text]
                )
                if tf_example:
                    writer.write(tf_example.SerializeToString())


flags = tf.app.flags
flags.DEFINE_string('json', '', 'Path to the JSON input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('images_dir', '', 'Path to images')
flags.DEFINE_string('label_map', '', 'Path to label the label map file')

flags.mark_flag_as_required('json')
flags.mark_flag_as_required('output_path')
flags.mark_flag_as_required('images_dir')
flags.mark_flag_as_required('label_map')

FLAGS = flags.FLAGS


def main(_):
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map)

    output_path = FLAGS.output_path

    examples = pd.read_json(FLAGS.json, lines=True)

    create_tf_record(examples, label_map_dict, FLAGS.images_dir, output_path)


if __name__ == '__main__':
    tf.app.run()
