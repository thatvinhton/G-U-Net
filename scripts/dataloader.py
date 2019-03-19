import os
import tensorflow as tf
import scipy.io as sio
from tensorflow.python.ops import control_flow_ops

IMG_MEANS = [143.166, 128.085, 143.166]


# These code learned from https://github.com/warmspringwinds/tf-image-segmentation/blob/master/tf_image_segmentation/utils/augmentation.py
def flip_left_right(img, ann):
    thres = tf.random_uniform(shape=[], maxval=2, dtype=tf.int32)

    flip_img = control_flow_ops.cond(pred=tf.equal(thres, 0),
                                     fn1=lambda: tf.image.flip_left_right(img),
                                     fn2=lambda: img)

    flip_ann = control_flow_ops.cond(pred=tf.equal(thres, 0),
                                     fn1=lambda: tf.image.flip_left_right(ann),
                                     fn2=lambda: ann)

    return flip_img, flip_ann


def flip_up_down(img, ann):
    thres = tf.random_uniform(shape=[], maxval=2, dtype=tf.int32)

    flip_img = control_flow_ops.cond(pred=tf.equal(thres, 0),
                                     fn1=lambda: tf.image.flip_up_down(img),
                                     fn2=lambda: img)

    flip_ann = control_flow_ops.cond(pred=tf.equal(thres, 0),
                                     fn1=lambda: tf.image.flip_up_down(ann),
                                     fn2=lambda: ann)

    return flip_img, flip_ann


def rotate_90(img, ann):
    num_rotate = tf.random_uniform(shape=[], maxval=4, dtype=tf.int32)

    img = control_flow_ops.cond(pred=tf.greater(num_rotate, 0),
                                fn1=lambda: tf.image.rot90(img),
                                fn2=lambda: img)

    ann = control_flow_ops.cond(pred=tf.greater(num_rotate, 0),
                                fn1=lambda: tf.image.rot90(ann),
                                fn2=lambda: ann)

    img = control_flow_ops.cond(pred=tf.greater(num_rotate, 1),
                                fn1=lambda: tf.image.rot90(img),
                                fn2=lambda: img)

    ann = control_flow_ops.cond(pred=tf.greater(num_rotate, 1),
                                fn1=lambda: tf.image.rot90(ann),
                                fn2=lambda: ann)

    img = control_flow_ops.cond(pred=tf.greater(num_rotate, 2),
                                fn1=lambda: tf.image.rot90(img),
                                fn2=lambda: img)

    ann = control_flow_ops.cond(pred=tf.greater(num_rotate, 2),
                                fn1=lambda: tf.image.rot90(ann),
                                fn2=lambda: ann)

    return img, ann


# https://github.com/awslabs/deeplearning-benchmark/blob/master/tensorflow/inception/inception/image_processing.py
def distort_1(image):
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.05)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

    return image


def distort_2(image):
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.05)

    return image


def distort_color(img, ann):
    thres = tf.random_uniform(shape=[], maxval=2, dtype=tf.int32)

    img = control_flow_ops.cond(pred=tf.equal(thres, 0), fn1=lambda: distort_1(img), fn2=lambda: distort_2(img))

    return img, ann


def do_augmentation(img, ann):
    # Flip & rotation
    img, ann = flip_up_down(img, ann)
    img, ann = flip_left_right(img, ann)
    img, ann = rotate_90(img, ann)

    # Distort color
    img = img / 255
    img, ann = distort_color(img, ann)
    img = img * 255

    return img, ann


def read_img_ann_list(root_dir):
    img_list = sorted([f for f in os.listdir(os.path.join(root_dir, 'img')) if os.path.isfile(os.path.join(root_dir, 'img', f))])
    ann_list = sorted([f for f in os.listdir(os.path.join(root_dir, 'ann')) if os.path.isfile(os.path.join(root_dir, 'ann', f))])

    img_list = [os.path.join(root_dir, 'img', f) for f in img_list]
    ann_list = [os.path.join(root_dir, 'ann', f) for f in ann_list]

    print(len(img_list))
    print(len(ann_list))

    return img_list, ann_list


def random_crop(img, ann, crop_size):
    combined = tf.concat(values=[img, tf.cast(ann, tf.float32)], axis=-1)

    crop_img_ann = tf.random_crop(combined, [crop_size[0], crop_size[1], 4])

    new_img, new_ann = tf.split(crop_img_ann, [3, 1], axis=-1)

    new_ann = tf.cast(new_ann, tf.int64)

    return new_img, new_ann


def crop_valid(img, ann, crop_size):
    combined = tf.concat(values=[img, tf.cast(ann, tf.float32)], axis=-1)

    # crop_img_ann = tf.random_crop(combined, [crop_size[0], crop_size[1], 4])
    crop_img_ann = tf.image.crop_to_bounding_box(combined, 0, 0, crop_size[0], crop_size[1])

    new_img, new_ann = tf.split(crop_img_ann, [3, 1], axis=-1)

    new_ann = tf.cast(new_ann, tf.int64)

    return new_img, new_ann


def read_img_from_disk(input_queue, input_size, crop_size, is_training):

    img_contents = tf.read_file(input_queue[0])
    ann_contents = tf.read_file(input_queue[1])

    img = tf.image.decode_jpeg(img_contents, channels=3)
    img = tf.cast(img, dtype=tf.float32)
    img = tf.reshape(img, shape=(input_size[0], input_size[1], 3))
    img -= IMG_MEANS

    ann = tf.image.decode_jpeg(ann_contents, channels=3)
    ann = tf.argmax(ann, axis=-1)
    ann = tf.reshape(ann, shape=(input_size[0], input_size[1], 1))

    # img, ann = random_crop(img, ann, crop_size)

    if is_training:
        img, ann = random_crop(img, ann, crop_size)
        img, ann = do_augmentation(img, ann)
    else:
        img, ann = crop_valid(img, ann, crop_size)

    ann = tf.squeeze(ann)

    return img, ann


class DataLoader(object):

    def __init__(self, root_dir, input_size, crop_size, is_training, is_validating, is_testing, coord):
        self.input_size = input_size
        self.crop_size = crop_size
        self.is_training = is_training
        self.is_validating = is_validating
        self.is_testing = is_testing

        if self.is_training:
            self.root_dir = os.path.join(root_dir, 'train')
        if self.is_testing:
            self.root_dir = os.path.join(root_dir, 'test')
        if self.is_validating:
            self.root_dir = os.path.join(root_dir, 'valid')

        self.coord = coord

        self.img_list, self.ann_list = read_img_ann_list(self.root_dir)
        self.images = tf.convert_to_tensor(self.img_list, dtype=tf.string)
        self.annos = tf.convert_to_tensor(self.ann_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.images, self.annos], shuffle=is_training)

        self.image, self.anno = read_img_from_disk(self.queue, self.input_size, self.crop_size, is_training)

    def dequeue(self, num_elements):

        image_batch, anno_batch = tf.train.batch([self.image, self.anno], num_elements)

        return image_batch, anno_batch
