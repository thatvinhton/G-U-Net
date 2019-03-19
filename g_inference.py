from __future__ import print_function
import argparse
import os
import sys
import time
from keras import backend as K

from PIL import Image
import tensorflow as tf
import numpy as np

from scripts.g_model import G_UNetResidual_Ext2, G_UNetResidual

NUM_CLASSES = 3
INPUT_SIZE = '128,128'
RESULT_DIR = './result'

IMG_MEANS = [143.166, 128.085, 143.166]

def get_arguments():
    parser = argparse.ArgumentParser(description='Inference U-Net')

    parser.add_argument("--img-link", type=str, required=True, help='Path to image dir/image link to inference.')
    parser.add_argument('--checkpoints', type=str, required=True, help='Path to restore weights.')
    parser.add_argument('--num-classes', type=int, default=NUM_CLASSES, help='Number of classes to preedict (including background).')
    parser.add_argument('--input-size', type=str, default=INPUT_SIZE, help='Comma-separated string with height and width of images.')
    parser.add_argument('--result-dir', type=str, default=RESULT_DIR)

    return parser.parse_args()

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print('Restored model parameters from {}'.format(ckpt_path))

def create_patch128(img_link):
    img = Image.open(img_link)
    height, width, _ = np.array(img).shape
    img = 1.0 * np.array(img)
    img = np.pad(img, ((128,128), (128, 128), (0, 0)), 'symmetric')
    img -= IMG_MEANS

    patch_list = []
    position_list = []

    num_rows = (height - 1) // 62 + 1
    num_cols = (width - 1) // 62 + 1

    for i in range(num_rows):
        for j in range(num_cols):
            x = 62 * i + 128 - 33
            y = 62 * j + 128 - 33

            patch = img[x:x + 128, y:y + 128, :]

            top = i * 62
            left = j * 62
            bot = min(height, (i + 1) * 62)
            right = min(width, (j + 1) * 62)

            position_list.append((top, left, bot, right))

            patch_list.append(patch)

    return (height, width), patch_list, position_list


def combine128(output_list, dim, position):
    result = np.zeros(shape=(dim[0], dim[1], 3), dtype=np.float32)

    for ind in range(len(output_list)):
        output = output_list[ind]
        top = position[ind][0]
        left = position[ind][1]
        bot = position[ind][2]
        right = position[ind][3]
        temp_output = output
        result[top:bot, left:right, :] = temp_output[33:33 +(bot-top), 33:33 + (right-left), :]

    return result

def run_infer(sess, image_holder, output, patch):
    patch = np.array(patch)
    patch_res_list = sess.run([output], feed_dict={image_holder: patch})
    res = np.array(patch_res_list[0])
    return res


def infer(sess, image_holder, output, img_link):
    dim, patch_list, position = create_patch128(img_link)

    ok = False
    num_repeat = 4 - (len(patch_list) % 4)
    if num_repeat == 4:
        num_repeat = 0
    for i in range(num_repeat):
        patch_list.append(patch_list[len(patch_list) - 1])
        ok = True

    output_patch_list = []
    for id in range(0, len(patch_list), 4):
        patch_res = run_infer(sess, image_holder, output, patch_list[id:id+4])
        for i in range(4):
            output_patch_list.append(patch_res[i])

    if ok:
        output_patch_list = output_patch_list[:-num_repeat]

    final_result = combine128(output_patch_list, dim, position)

    return final_result


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")

    return graph


def main():
    args = get_arguments()

    h, w= map(int, args.input_size.split(','))
    input_size = (h, w)

    # Image holder
    image_holder = tf.placeholder(tf.float32, shape=(4, h, w, 3))

    # Create network
    with tf.variable_scope('', reuse=False):
        net = G_UNetResidual_Ext2({'data': image_holder}, is_training=False, num_classes=args.num_classes)

    with tf.variable_scope('inference_result'):
        raw_output = net.getOutput()
        normalized_output = tf.squeeze(tf.nn.softmax(raw_output))

    restore_var = tf.global_variables()
    print(restore_var)

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.keras.backend.get_session()
    K.set_session(sess)
    K.set_learning_phase(0)

    global_init = tf.global_variables_initializer()
    sess.run(global_init)

    ckpt = tf.train.get_checkpoint_state(args.checkpoints)
    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=restore_var)
        load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found.')

    if True:

        if os.path.isfile(args.img_link):
            start_time = time.time()

            result = infer(sess, image_holder, normalized_output, args.img_link)

            img_res = Image.fromarray((result * 255).astype(np.uint8))
            img_res.save('output.png')

            print('Inference time: %.3lf secs' % (time.time() - start_time))
        else:
            file_list = sorted([f for f in os.listdir(args.img_link) if os.path.isfile(os.path.join(args.img_link, f))])

            res_dir = args.result_dir

            if not os.path.exists(res_dir):
                os.makedirs(res_dir)

            sum_time = 0
            for file_name in file_list:
                print(file_name)
                start_time = time.time()
                result = infer(sess, image_holder, normalized_output, os.path.join(args.img_link, file_name))

                img_res = Image.fromarray((result * 255).astype(np.uint8))
                img_res.save(os.path.join(res_dir, file_name))

                end_time = time.time()
                sum_time = sum_time + end_time - start_time
                print('Finished inference in %.3lf secs' % (end_time - start_time))

            print('Avg inference time: %.3lf secs' % (sum_time / len(file_list)))


if __name__ == '__main__':
    main()
