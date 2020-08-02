import os
import sys
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from config import cfg
from utils import load_data
from capsNet import CapsNet
from PIL import Image
from sklearn import metrics
import pandas as pd

def save_to():
    if not os.path.exists(cfg.results):
        os.mkdir(cfg.results)
    if cfg.is_training:
        loss = cfg.results + '/loss_' + str(cfg.k_fold) + '.csv'
        train_acc = cfg.results + '/train_acc_' + str(cfg.k_fold) + '.csv'
        val_acc = cfg.results + '/val_acc_' + str(cfg.k_fold) + '.csv'

        if os.path.exists(val_acc):
            os.remove(val_acc)
        if os.path.exists(loss):
            os.remove(loss)
        if os.path.exists(train_acc):
            os.remove(train_acc)

        fd_train_acc = open(train_acc, 'w')
        fd_train_acc.write('step,train_acc\n')
        fd_loss = open(loss, 'w')
        fd_loss.write('step,loss\n')
        fd_val_acc = open(val_acc, 'w')
        fd_val_acc.write('step,val_acc\n')
        return(fd_train_acc, fd_loss, fd_val_acc)
    else:
        test_acc = cfg.results + '/test_acc' + str(cfg.k_fold) + '.csv'
        if os.path.exists(test_acc):
            os.remove(test_acc)
        fd_test_acc = open(test_acc, 'w')
        fd_test_acc.write('test_acc\n')
        return(fd_test_acc)


def train(model, supervisor, num_label):
    start_time = time.time()
    trX, trY, num_tr_batch, valX, valY, num_val_batch = load_data(cfg.dataset, cfg.batch_size, is_training=True)
    print("num_tr_batch = " + str(num_tr_batch))
    Y = valY[:num_val_batch * cfg.batch_size].reshape((-1, 1))

    fd_train_acc, fd_loss, fd_val_acc = save_to()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    print("--- %s seconds ---" % (time.time() - start_time))

    with supervisor.managed_session(config=config) as sess:
        print("\nNote: all of results will be saved to directory: " + cfg.results)
        for epoch in range(cfg.epoch):
            print("Training for epoch %d/%d:" % (epoch, cfg.epoch))
            if supervisor.should_stop():
                print('supervisor stoped!')
                break
            for step in tqdm(range(num_tr_batch), total=num_tr_batch, ncols=70, leave=False, unit='b'):
                start = step * cfg.batch_size
                end = start + cfg.batch_size
                global_step = epoch * num_tr_batch + step

                if global_step % cfg.train_sum_freq == 0:
                    _, loss, train_acc, summary_str = sess.run([model.train_op, model.total_loss, model.accuracy, model.train_summary])
                    assert not np.isnan(loss), 'Something wrong! loss is nan...'
                    supervisor.summary_writer.add_summary(summary_str, global_step)

                    fd_loss.write(str(global_step) + ',' + str(loss) + "\n")
                    fd_loss.flush()
                    fd_train_acc.write(str(global_step) + ',' + str(train_acc / cfg.batch_size) + "\n")
                    fd_train_acc.flush()
                else:
                    sess.run(model.train_op)

                if cfg.val_sum_freq != 0 and (global_step) % cfg.val_sum_freq == 0:
                    val_acc = 0
                    for i in range(num_val_batch):
                        start = i * cfg.batch_size
                        end = start + cfg.batch_size
                        acc = sess.run(model.accuracy, {model.X: valX[start:end], model.labels: valY[start:end]})
                        val_acc += acc
                    val_acc = val_acc / (cfg.batch_size * num_val_batch)
                    fd_val_acc.write(str(global_step) + ',' + str(val_acc) + '\n')
                    fd_val_acc.flush()

            if (epoch + 1) % cfg.save_freq == 0:
                supervisor.saver.save(sess, cfg.logdir + '/model_epoch_%04d_step_%02d' % (epoch, global_step))

        fd_val_acc.close()
        fd_train_acc.close()
        fd_loss.close()
        print("--- %s seconds ---" % (time.time() - start_time))


def evaluation(model, supervisor, num_label):
    teX, teY, num_te_batch = load_data(cfg.dataset, cfg.batch_size, is_training=False)
    fd_test_acc = save_to()
    with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        supervisor.saver.restore(sess, tf.train.latest_checkpoint(cfg.logdir))
        tf.logging.info('Model restored!')

        test_acc = 0
        test_label = []
        test_argmax_idx = []
        for i in tqdm(range(num_te_batch), total=num_te_batch, ncols=70, leave=False, unit='b'):
            start = i * cfg.batch_size
            end = start + cfg.batch_size
            acc, label, argmax_idx = sess.run([model.accuracy, model.labels, model.argmax_idx], {model.X: teX[start:end], model.labels: teY[start:end]})
            print(label)
            test_label.extend(label)
            print(argmax_idx)
            test_argmax_idx.extend(argmax_idx)
            test_acc += acc
        print(num_te_batch)
        # Print the confusion matrix
        print(metrics.confusion_matrix(test_label, test_argmax_idx))

        # Print the precision and recall, among other metrics
        print(metrics.classification_report(test_label, test_argmax_idx, digits=2))
        report = metrics.classification_report(test_label, test_argmax_idx, digits=2, output_dict=True)
        df = pd.DataFrame(report).transpose()
        df.to_csv(cfg.results + '/confusion_matrix_' + str(cfg.k_fold) + '.csv', encoding='utf-8')

        test_acc = test_acc / (cfg.batch_size * num_te_batch)
        fd_test_acc.write(str(test_acc))
        fd_test_acc.close()
        print('Test accuracy has been saved to ' + cfg.results + '/test_acc.csv')

def evaluation_each(model, supervisor, num_label):
    aksara=["A","BA","CA","DA","GA","HA","I","JA","KA","LA","MA","NA","NGA","NYA","PA","PANELENG","PANEULEUNG","PANGHULU","PANGLAYAR","PANOLONG","PANYUKU","PATEN","RA","SA","TA","U","WA","YA"]
    
    script_dir = os.path.abspath('')
    directory_path = os.path.join(script_dir,"data","sunda_kuno","train-test_image")

    img_np = np.array([])
    img_label = np.array([])

    # abs_file_path = os.path.join(directory_path, "A_%s.png" % (str(y)))
    abs_file_path = os.path.join(directory_path, "A_1.png")
    img = Image.open(abs_file_path)
    img = np.array(img)
    img = img[:, :]

    for y in range(1, 64):

        img_np = np.append(img_np,img)
        img_label = np.append(img_label,0)

    abs_file_path = os.path.join(directory_path, "%s.png" % (str(cfg.aksara_test)))
    img = Image.open(abs_file_path)
    img = np.array(img)
    img = img[:, :]

    final_np = np.append(img_np,img)

    label_num=0
    test_acc=0
    for i in range(0,len(aksara)):
        final_label = np.append(img_label,i)
        teX = final_np.reshape((64, 28, 28, 1)).astype(np.float32)
        teY = final_label.reshape((64)).astype(np.int32)
        # fd_test_acc = save_to()
        with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            supervisor.saver.restore(sess, tf.train.latest_checkpoint(cfg.logdir))
            tf.logging.info('Model restored!')

            test_acc = sess.run(model.accuracy, {model.X: teX, model.labels: teY})
            print(str(aksara[i]) + ": " + str(test_acc/64))
            if test_acc/64 == 1:
                label_num=i
                break
    print(aksara[label_num])



    # abs_file_path = os.path.join(os.path.join(script_dir,"data","sunda_kuno","train-test_image"), "A_36.png")
    # img = Image.open(abs_file_path)
    # img = np.array(img)

    # teX = img[:, :]
    # teY = 0
    # num_te_batch = 1
    # teX = teX.reshape((1, 28, 28, 1)).astype(np.float32)
    # teY = teY.reshape((1)).astype(np.int32)
    # print(teX)
    # print(teY)
    # teX, teY, num_te_batch = load_data(cfg.dataset, cfg.batch_size, is_training=False)

    # fd_test_acc = save_to()
    # with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    #     supervisor.saver.restore(sess, tf.train.latest_checkpoint(cfg.logdir))
    #     tf.logging.info('Model restored!')

    #     test_acc = 0
    #     # for i in tqdm(range(num_te_batch), total=num_te_batch, ncols=70, leave=False, unit='b'):
    #     #     start = i * cfg.batch_size
    #     #     end = start + cfg.batch_size
    #     acc = sess.run(model.accuracy, {model.X: teX, model.labels: teY})
    #     # test_acc += acc
    #     print(acc/64)
        # test_acc = test_acc
        # fd_test_acc.write(str(test_acc))
        # fd_test_acc.close()
        # print('Test accuracy has been saved to ' + cfg.results + '/test_acc.csv')

def main(_):
    tf.logging.info(' Loading Graph...')
    num_label = 28
    model = CapsNet()
    tf.logging.info(' Graph loaded')

    sv = tf.train.Supervisor(graph=model.graph, logdir=cfg.logdir, save_model_secs=0)

    if cfg.is_training:
        tf.logging.info(' Start training...')
        train(model, sv, num_label)
        tf.logging.info('Training done')
    else:
        if cfg.aksara_test=='':
            evaluation(model, sv, num_label)
        else:
            evaluation_each(model, sv, num_label)

if __name__ == "__main__":
    tf.app.run()
