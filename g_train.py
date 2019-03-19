import argparse
import os
import sys
import time

import tensorflow as tf
import numpy as np

from scripts.g_model import G_UNetResidual_Ext2
from scripts.dataloader import DataLoader
from scripts.tensorboard_logging import Logger
from keras import backend as K

from niftynet.layer.loss_segmentation import generalised_dice_loss
from PIL import Image
from jaccard import calAJI, post_process

BATCH_SIZE = 2
DATA_DIRECTORY = './data/patchDir_aug_2_pixel_for_overlap'
INPUT_SIZE = '256,256'
CROP_SIZE='256,256'
LEARNING_RATE = 5e-5
NUM_CLASSES = 3
NUM_STEPS = 1000000
RANDOM_SEED = 1234
RESTORE_FROM = './'
SNAPSHOT_DIR_SUFFIX = 'model'
SAVE_PRED_EVERY = 500
NUM_STEP_PER_EPOCH = 34500

LOGS_DIR_SUFFIX = 'logs'

LOGS_ROOT = './expers'

CLASS_WEIGHTS = tf.constant([1, 1, 1])

IMG_MEANS = [143.166, 128.085, 143.166]


UPDATE_OPS_COLLECTION = 'resnet_update_ops'

OPENING_FULL_MASK_ITER = 1
OPENING_CELL_SEED_ITER = 2
THRES_SEED = 0.85
FINAL_EXPAND_STEP = 1


def toOneHot(label):
    res = np.zeros(shape=(label.shape[0], label.shape[1], 3))

    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            res[i, j, label[i, j]] = 1

    return res

def cal_AJI_batch(predicted, gt, batch_size):
    res = 0

    for i in range(batch_size):
        predicted_mat = post_process(predicted[i], opening_full_mask_iter=OPENING_FULL_MASK_ITER,
                                     opening_cell_seed_iter=OPENING_CELL_SEED_ITER, thres=THRES_SEED,
                                     final_expanding_iter=FINAL_EXPAND_STEP)

        gt_mat = post_process(toOneHot(gt[i]), opening_full_mask_iter=0, opening_cell_seed_iter=0, thres=THRES_SEED,
                              final_expanding_iter=FINAL_EXPAND_STEP)

        res = res + calAJI(predicted_mat, gt_mat)

    return res / batch_size

def get_arguments():
    parser = argparse.ArgumentParser(description="UNet experiments.")
    parser.add_argument("--exper-name", type=str, default='', required=True,
                        help="Experiment name")

    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--crop-size", type=str, default=CROP_SIZE,
                        help="Comma-separated string with height and width of cropped images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")

    return parser.parse_args()


def save(saver, sess, logdir, step):
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)

   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')


def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def addNameToTensor(tensor, name):
    return tf.identity(tensor, name=name)


def main():
    """Create the model and start the training."""
    args = get_arguments()

    logs_dir = os.path.join(LOGS_ROOT, args.exper_name, LOGS_DIR_SUFFIX)
    snap_dir = os.path.join(LOGS_ROOT, args.exper_name, SNAPSHOT_DIR_SUFFIX)

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    c_h, c_w = map(int, args.crop_size.split(','))
    crop_size = (c_h, c_w)

    tf.set_random_seed(args.random_seed)

    # Coordinator for threads
    coord = tf.train.Coordinator()

    with tf.name_scope("create_inputs"):
        reader = DataLoader(args.data_dir, input_size, crop_size, True, False, False, coord)
        image_batch, label_batch = reader.dequeue(args.batch_size)
        addNameToTensor(image_batch, 'input')

    with tf.name_scope("validating_input"):
       validate_reader = DataLoader(args.data_dir, input_size, crop_size, False, True, False, coord)
       image_validate_batch, label_validate_batch = validate_reader.dequeue(args.batch_size)

    # for training
    with tf.variable_scope(''):
        net = G_UNetResidual_Ext2({'data': image_batch}, is_training=True, num_classes=args.num_classes)

    for layer in net.layers:
        print(layer)
        print(net.layers[layer].shape)

    # tensors = [n.name for n in tf.get_default_graph().as_graph_def().node]
    # for tensor in tensors[:2000]:
    #     print(tensor)

    total_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
#        print(variable)
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_params += variable_parameters
    print("Number of trainable parameters: %d" % (total_params))
#    return

    # for validation
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
       net_val = G_UNetResidual_Ext2({'data': image_validate_batch}, is_training=False, num_classes=args.num_classes)

    with tf.variable_scope('training_output'):
        # output training
        logits = net.getOutput()

    addNameToTensor(logits, 'output')

    with tf.variable_scope('validation_output'):
       # output validation
       logits_validation = net_val.getOutput()

    restore_var = [v for v in tf.global_variables()]

    with tf.variable_scope('training_loss'):
        train_weights = tf.gather(CLASS_WEIGHTS, label_batch)

        # loss for training
        ce_loss = tf.losses.sparse_softmax_cross_entropy(label_batch, logits, train_weights)
        ce_reduced_loss = tf.reduce_mean(ce_loss)

        dice_loss = generalised_dice_loss(tf.nn.softmax(logits), label_batch)

        train_loss = (ce_reduced_loss + dice_loss)

    with tf.variable_scope('validation_loss'):
       valid_weights = tf.gather(CLASS_WEIGHTS, label_validate_batch)

       # loss for validation
       ce_loss_validation = tf.losses.sparse_softmax_cross_entropy(label_validate_batch, logits_validation, valid_weights)
       ce_reduced_loss_validation = tf.reduce_mean(ce_loss_validation)

       dice_loss_validation = generalised_dice_loss(tf.nn.softmax(logits_validation),label_validate_batch)
       valid_loss = (ce_reduced_loss_validation + dice_loss_validation)

    with tf.variable_scope('accuracy'):
        # accuracy
        preds = tf.argmax(logits_validation, axis=-1)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, label_validate_batch), tf.double))

    with tf.variable_scope('probability'):
        probs = tf.nn.softmax(logits_validation)

    with tf.variable_scope('train'):
        batch_ops = tf.get_collection(UPDATE_OPS_COLLECTION)
        old_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # print(batch_ops)
        # print(old_update_ops)

        update_ops = batch_ops + old_update_ops

        with tf.control_dependencies(update_ops):
            # training optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate = args.learning_rate)
            train_op = optimizer.minimize(train_loss, global_step=tf.train.get_global_step())

    init = tf.global_variables_initializer()

    sess = tf.keras.backend.get_session()
    K.set_session(sess)
    K.set_learning_phase(1)
    sess.run(init)

    train_logger = Logger(logs_dir + '/train')
    valid_logger = Logger(logs_dir + '/valid')

    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)

    # initialize weight
    ckpt = tf.train.get_checkpoint_state(snap_dir)
    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=restore_var)
        load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found.')
        load_step = 0

    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    min_val_loss = 1e9
    best_acc = 0

    print(K.learning_phase())

    for step in range(args.num_steps):
        realStep = step + load_step

        start_time = time.time()

        loss_value, _ = sess.run([train_loss, train_op], feed_dict={net.update_bn: True})

        train_logger.log_scalar('loss', loss_value, realStep)

        if realStep % args.save_pred_every == 0:

            # train_output = np.uint8(batch_probs[0] * 255)
            # img = Image.fromarray(train_output)
            # img.save('temp_output.png')
            print('Validation')
            numValidateImg = len(os.listdir(os.path.join(args.data_dir, 'valid', 'img')))
            numStep = int(numValidateImg / args.batch_size)
            loss_validation_test = 0
            accuracy = 0

            for step_val in range(numStep):
                predicted, batch_prob, gt, l, ac = sess.run([logits_validation, probs, label_validate_batch, valid_loss, acc])
                if step_val == 0:
                    valid_output = np.uint8(batch_prob[0] * 255)
                    img = Image.fromarray(valid_output)
                    img.save('temp_valid_output.png')

                    gt_output = np.uint8(toOneHot(gt[0]) * 255)
                    img = Image.fromarray(gt_output)
                    img.save('temp_gt_output.png')

                loss_validation_test = loss_validation_test + l / numStep
                # accuracy = accuracy + ac / numStep
                accuracy = accuracy + cal_AJI_batch(predicted, gt, args.batch_size) / numStep

            print('Validation result: loss = %.5f, acc = %.5f' % (loss_validation_test, accuracy))

            if accuracy > best_acc:
                print('Update model: previous loss: %4.lf, new loss: %.4lf, step: %d' % (min_val_loss, loss_validation_test, realStep))
                min_val_loss = loss_validation_test
                best_acc = accuracy
                save(saver, sess, snap_dir, realStep)

            valid_logger.log_scalar('loss', loss_validation_test,  realStep)
            valid_logger.log_scalar('acc', accuracy, realStep)

        duration = time.time() - start_time

        print('step %d \t loss = %.5f, (%.3f sec/step)' % (realStep, loss_value, duration))

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main()
