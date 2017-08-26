import os, argparse, sys

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.tools.inspect_checkpoint import *

from models.VGG.vgg import *
from models.inception_resnet_v2.inception_resnet_v2 import *
import data

slim = tf.contrib.slim
layers = tf.contrib.layers
framework = tf.contrib.framework

VGG_MEAN = [123.68, 116.78, 103.94]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file', default='./models/VGG/vgg_16.ckpt', help='Path to the latest checkpoint file')
    parser.add_argument('--dataset', default='NTURGBD', help='Which dataset to load for Action Recognition')
    parser.add_argument('--dataset_dir', default='/home/procastinator/ActRec-skeleton/image_data', help='Path to base directory for the TFRecord files')
    parser.add_argument('--splits_dir', default='/home/procastinator/NTU_data', help='Path to base directory for the TFRecord files')
    parser.add_argument('--num_epochs', default='4', type=int, help='Path to base directory for the TFRecord files')
    parser.add_argument('--batch_size', default='24', type=int, help='Path to base directory for the TFRecord files')
    parser.add_argument('--arch', default='vgg_16', help='Which architecture to use for the model')
    parser.add_argument('--split_num', default='1', help='Which architecture to use for the model')
    parser.add_argument('--lr', default=0.001, type=float, help='Which architecture to use for the model')
    parser.add_argument('--train', default=1, type=int, help='Which architecture to use for the model')

    args = parser.parse_args()

    dataset_fn = getattr(data, args.dataset)
    data = dataset_fn(args.dataset_dir, args.splits_dir, args.num_epochs, args.batch_size)

    if args.train:
        split_file = os.path.join(data.splits_dir, data.train_split_files[args.split_num])

    rgb_image_files, labels = data._read_labeled_image_list(split_file)

    rgb_image_files = ops.convert_to_tensor(rgb_image_files, dtype=dtypes.string)
    labels = ops.convert_to_tensor(labels, dtype=dtypes.int32)

    input_queue = tf.train.slice_input_producer(
                        [rgb_image_files, labels],
                        num_epochs = data.num_epochs,
                        shuffle = True)
    rgb_image, label = data._read_images_from_disk(input_queue)
    if args.arch == 'vgg_16':
        rgb_image = tf.image.resize_images(rgb_image, (224, 224))
    elif args.arch == 'inception_resnet_v2':
        rgb_image = tf.image.resize_images(rgb_image, (299, 299))


    rgb_image_loader, label_loader = \
                        tf.train.shuffle_batch(
                                [rgb_image, label],
                                batch_size = data.batch_size,
                                capacity = 5 * data.batch_size,
                                min_after_dequeue = data.batch_size)

    print rgb_image_loader.get_shape(), label_loader.get_shape()

    if args.arch == 'vgg_16':
        arg_scope = vgg_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, _ = vgg_16(rgb_image_loader, num_classes=data.num_classes)
        var_list = framework.get_variables_to_restore(exclude=['vgg_16/fc8'])
        with tf.variable_scope('fc8'):
            fc8_vars = framework.get_variables('vgg_16/fc8')
            fc8_init = tf.variables_initializer(fc8_vars)
    elif args.arch == 'inception_resnet_v2':
        arg_scope = inception_resnet_v2_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, _ = inception_resnet_v2(rgb_image_loader, num_classes=data.num_classes)
        t_vars = tf.trainable_variables()
        var_list = []
        Logits = []
        for var in t_vars:
            if not ("Logits/Logits" in var.name
                    or "AuxLogits/Logits" in var.name):
                var_list.append(var)
            else:
                Logits.append(var)
        logits_init = tf.variables_initializer(Logits)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    #config = tf.ConfigProto(device_count = {'GPU': 0})
    config = tf.ConfigProto()

    # Import the variables for 3D model and get trainable variables
    with tf.Session(config=config) as sess:

        labels = tf.to_int64(label_loader)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits, name='xentropy')

        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        tf.summary.scalar('loss', loss)

        learning_rate = tf.train.exponential_decay(
            args.lr,
            global_step * args.batch_size,
            200000,
            0.9,
            staircase=True)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)

        print '[INFO] Initialize all global variables and load checkpoint'
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        if not args.checkpoint_file == '':
            print 'HERE'
            saver = tf.train.Saver(var_list)
            saver.restore(sess, args.checkpoint_file)
        else:
            sess.run(tf.variables_initializer(var_list))

        if args.arch == 'vgg_16':
            sess.run(fc8_init)
        else:
            sess.run(logits_init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        tf.add_to_collection('global_step', global_step)
        tf.add_to_collection('loss', loss)
        tf.add_to_collection('predictions', logits)
        tf.add_to_collection('input_batch', rgb_image_loader)
        tf.add_to_collection('labels', label_loader)
        tf.add_to_collection('train_op', train_op)

        saver_new = tf.train.Saver()
        saver_new.save(sess, 'vgg_16_nturgbd')

#        print '[INFO] Start training for 10000 steps'
#        for step in range(10000):
#            print('Running step %d' % (step + 1))
#            _, loss_value = sess.run([train_op, loss])
#            if (step + 1) % 10 == 0:
#                print('Step %d: loss = %.2f' % (step + 1, loss_value))

        coord.request_stop()
        coord.join(threads)
