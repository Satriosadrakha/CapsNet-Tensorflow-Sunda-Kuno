import os
import scipy
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter, ImageEnhance
import glob
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from config import cfg

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import cv2 as cv

def load_mnist(batch_size, is_training=True):
    path = os.path.join('data', 'mnist')
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((60000)).astype(np.int32)

        trX = trainX[:55000] / 255.
        trY = trainY[:55000]

        valX = trainX[55000:, ] / 255.
        valY = trainY[55000:]

        num_tr_batch = 55000 // batch_size
        num_val_batch = 5000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch

def load_fashion_mnist(batch_size, is_training=True):
    path = os.path.join('data', 'fashion-mnist')
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((60000)).astype(np.int32)

        trX = trainX[:55000] / 255.
        trY = trainY[:55000]

        valX = trainX[55000:, ] / 255.
        valY = trainY[55000:]

        num_tr_batch = 55000 // batch_size
        num_val_batch = 5000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch

def load_sunda_kuno(batch_size, is_training=True):
    if is_training:
        if cfg.k_fold==0:
            trainX, trainY, sumTrain, sumVal = get_sunda_kuno(is_training)
            
            trX = trainX[:sumTrain] / 255.
            trY = trainY[:sumTrain]

            valX = trainX[sumTrain:, ] / 255.
            valY = trainY[sumTrain:]
        else:
            trX, valX, trY, valY = get_sunda_kuno_crossval(is_training)
            
            trX = trX / 255.
            valX = valX / 255.
            
            sumTrain = len(trX)
            sumVal = len(valX)

        num_tr_batch = sumTrain // batch_size
        num_val_batch = sumVal // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        if cfg.k_fold==0:
            teX, teY, sumImg = get_sunda_kuno(False)
        else:
            teX, teY, sumImg = get_sunda_kuno_crossval(False)
        
        num_te_batch = sumImg // batch_size
        return teX / 255., teY, num_te_batch

def load_khmer(batch_size, is_training=True):
    if is_training:
        if cfg.k_fold==0:
            trainX, trainY, sumTrain, sumVal = get_khmer(is_training)
            
            trX = trainX[:sumTrain] / 255.
            trY = trainY[:sumTrain]

            valX = trainX[sumTrain:, ] / 255.
            valY = trainY[sumTrain:]
        else:
            trX, valX, trY, valY = get_khmer_crossval(is_training)
            
            trX = trX / 255.
            valX = valX / 255.
            
            sumTrain = len(trX)
            sumVal = len(valX)

        num_tr_batch = sumTrain // batch_size
        num_val_batch = sumVal // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        if cfg.k_fold==0:
            teX, teY, sumImg = get_khmer(False)
        else:
            teX, teY, sumImg = get_khmer_crossval(False)
        
        num_te_batch = sumImg // batch_size
        return teX / 255., teY, num_te_batch

def get_sunda_kuno_each(is_training,aksara,subtract=0):
    script_dir = os.path.abspath('')
    i=["A","BA","CA","DA","GA","HA","I","JA","KA","LA","MA","NA","NGA","NYA","PA","PANELENG","PANEULEUNG","PANGHULU","PANGLAYAR","PANOLONG","PANYUKU","PATEN","RA","SA","TA","U","WA","YA"]

    final_np = np.array([])
    final_label = np.array([])

    sumImg = 0
    directory_path = os.path.join(script_dir,"data","sunda_kuno","train-test_image",str(cfg.dataset_pre))
    count_image = len(glob.glob1(directory_path,"%s_*.png" % (i[aksara])))

    print("Aksara = " + str(i[aksara]))
    print("Count_image training = " + str(count_image))

    for y in range(1, count_image+1):
        abs_file_path = os.path.join(directory_path, "%s_%s.png" % (i[aksara],str(y)))
        img = Image.open(abs_file_path)
        img = np.array(img)
        img = img[:, :]
        final_np = np.append(final_np,img)
        final_label = np.append(final_label,aksara)
    sumImg = sumImg + count_image
    np.random.seed(42)

    if cfg.koropak28_test==False:
    	# print()
    	final_np = final_np.reshape((sumImg, 28, 28, 1)).astype(np.float32)
    	final_label = final_label.reshape((sumImg)).astype(np.int32)
    	
    	if sumImg < cfg.sampling_threshold:
    		idx = np.random.choice(sumImg,size=sumImg*2,replace=True)
    	else:
    		idx = np.random.choice(sumImg,size=sumImg-subtract,replace=False)
    	print("Idx = " + str(len(idx)))

    	data_train, data_test, labels_train, labels_test = train_test_split(final_np[idx,:], final_label[idx], test_size=0.30, random_state=42)

    	if is_training == True:
    		return(data_train,labels_train)
    	else:
    		data_val, data_test, labels_val, labels_test = train_test_split(data_test, labels_test, test_size=0.50, random_state=42)
    		return(data_val, data_test, labels_val, labels_test)

    else:
    	# final_np = final_np.reshape((sumImg, 28, 28, 1)).astype(np.float32)
    	# final_label = final_label.reshape((sumImg)).astype(np.int32)

    	# if sumImg < cfg.sampling_threshold:
    	# 	idx = np.random.choice(sumImg,size=sumImg*2,replace=True)
    	# else:
    	# 	idx = np.random.choice(sumImg,size=sumImg-subtract,replace=False)
    	# print("Idx = " + str(len(idx)))

    	# data_train = final_np[idx,:]
    	# labels_train = final_label[idx]

    	# final_np = np.array([])
    	# final_label = np.array([])

        # sumImg = 0


        directory_path = os.path.join(script_dir,"data","sunda_kuno","train-test_image","koropak_28")
        count_image = len(glob.glob1(directory_path,"%s_*.png" % (i[aksara])))
        print("Count_image koropak = " + str(count_image))

        for y in range(0, count_image):
            abs_file_path = os.path.join(directory_path, "%s_%s.png" % (i[aksara],str(y)))
            img = Image.open(abs_file_path)
            img = np.array(img)
            img = img[:, :]
            final_np = np.append(final_np,img)
            final_label = np.append(final_label,aksara)
        sumImg = sumImg + count_image

        directory_path = os.path.join(script_dir,"data","sunda_kuno","train-test_image","kawih_p")
        count_image = len(glob.glob1(directory_path,"%s_*.png" % (i[aksara])))
        print("Count_image kawih = " + str(count_image))

        for y in range(0, count_image):
            abs_file_path = os.path.join(directory_path, "%s_%s.png" % (i[aksara],str(y)))
            img = Image.open(abs_file_path)
            img = np.array(img)
            img = img[:, :]
            final_np = np.append(final_np,img)
            final_label = np.append(final_label,aksara)
        sumImg = sumImg + count_image

        print("SumImg = " + str(sumImg))

        final_np = final_np.reshape((sumImg, 28, 28, 1)).astype(np.float32)
        final_label = final_label.reshape((sumImg)).astype(np.int32)

        # if cfg.sampling_threshold!=0:
        #     if sumImg < 50:
        #         idx = np.random.choice(sumImg,size=50,replace=True)
        #     else:
        #         idx = np.random.choice(sumImg,size=50,replace=False)
        #     print("Idx = " + str(len(idx)))
        #     data_train, data_test, labels_train, labels_test = train_test_split(final_np[idx,:], final_label[idx], test_size=0.30, random_state=42)
        # else:
        #     data_train, data_test, labels_train, labels_test = train_test_split(final_np, final_label, test_size=0.30, random_state=42)
        # # idx = np.random.choice(sumImg,size=50,replace=True)

        data_train, data_test, labels_train, labels_test = train_test_split(final_np, final_label, test_size=0.30, random_state=42)
        print("Data train = " + str(len(data_train)))

        if cfg.gan==True:
            sumImgTrain = len(data_train)
            
            final_np = np.array([])
            final_label = np.array([])

            sumImgGAN = 0
            directory_path = os.path.join(script_dir,"data","sunda_kuno","GAN_generated_images","GAN_training_only")
            count_image = len(glob.glob1(directory_path,"%s_*.png" % (i[aksara])))

            for y in range(sumImg+1, sumImg+count_image+1):
                abs_file_path = os.path.join(directory_path, "%s_%s.png" % (i[aksara],str(y)))
                img = Image.open(abs_file_path)
                img = np.array(img)
                img = img[:, :]
                final_np = np.append(final_np,img)
                final_label = np.append(final_label,aksara)
            print("Generated images: " + str(count_image))
            sumImgGAN = sumImgGAN + count_image
            sumImgTrain = sumImgTrain + sumImgGAN

            final_np = final_np.reshape((sumImgGAN, 28, 28, 1)).astype(np.float32)
            final_label = final_label.reshape((sumImgGAN)).astype(np.int32)

            print("Final_np = " + str(final_np.shape))
            print("Data train to be appended = " + str(data_train.shape))

            data_train = np.append(data_train,final_np)
            labels_train = np.append(labels_train,final_label)

            data_train = data_train.reshape((sumImgTrain, 28, 28, 1)).astype(np.float32)
            labels_train = labels_train.reshape((sumImgTrain)).astype(np.int32)

            print("Data train akhir = " + str(len(data_train)))

        # if sumImg < cfg.sampling_threshold:
        #     idx = np.random.choice(sumImg,size=sumImg*2,replace=True)
        # else:
        #     idx = np.random.choice(sumImg,size=sumImg-subtract,replace=False)
        # print("Idx = " + str(len(idx)))

        # data_train, data_test, labels_train, labels_test = train_test_split(final_np[idx,:], final_label[idx], test_size=0.30, random_state=42)
        # data_train, data_test, labels_train, labels_test = train_test_split(final_np, final_label, test_size=0.30, random_state=42)
        data_val, data_test, labels_val, labels_test = train_test_split(data_test, labels_test, test_size=0.50, random_state=42)
        if cfg.save_train_test == True:
            for x in range(0, len(data_train)):
                # print(data_train[x,:,:,0].shape)
                # print(data_train[x,:,:,0])

                im = Image.fromarray(data_train[x,:,:,0])
                im = im.convert("L")
                im.save("imgs/training/"+i[aksara]+"-"+str(x)+".jpeg")
            for x in range(0, len(data_val)):
                im = Image.fromarray(data_val[x,:,:,0])
                im = im.convert("L")
                im.save("imgs/validation/"+i[aksara]+"-"+str(x)+".jpeg")
            for x in range(0, len(data_test)):
                im = Image.fromarray(data_test[x,:,:,0])
                im = im.convert("L")
                im.save("imgs/testing/"+i[aksara]+"-"+str(x)+".jpeg")

        if is_training == True:
            return(data_train,labels_train)
        else:
            return(data_val, data_test, labels_val, labels_test)

    	# data_test = final_np
    	# labels_test = final_label

    	# # data_train, data_val, labels_train, labels_val = train_test_split(data_train, labels_train, test_size=0.20, random_state=42)
    	# data_val, data_test, labels_val, labels_test = train_test_split(data_test, labels_test, test_size=0.50, random_state=42)
    	
    	# if is_training == True:
    	# 	return(data_train,labels_train)
    	# else:
    	# 	return(data_val, data_test, labels_val, labels_test)
     #    # data_val, data_test, labels_val, labels_test = train_test_split(data_test, labels_test, test_size=0.50, random_state=42)

    print(i[aksara] + " " + len(labels_train) + " " + len(labels_val) + " " + len(labels_test))
    # if is_training == True:
    #     return(data_train,labels_train)
    # else:
    #     data_val, data_test, labels_val, labels_test = train_test_split(data_test, labels_test, test_size=0.50, random_state=42)
    #     return(data_val, data_test, labels_val, labels_test)

def get_sunda_kuno_each_crossval(is_training,aksara,subtract=0):
    script_dir = os.path.abspath('')
    i=["A","BA","CA","DA","GA","HA","I","JA","KA","LA","MA","NA","NGA","NYA","PA","PANELENG","PANEULEUNG","PANGHULU","PANGLAYAR","PANOLONG","PANYUKU","PATEN","RA","SA","TA","U","WA","YA"]
    
    final_np = np.array([])
    final_label = np.array([])

    sumImg = 0
    directory_path = os.path.join(script_dir,"data","sunda_kuno","train-test_image",str(cfg.dataset_pre))
    count_image = len(glob.glob1(directory_path,"%s_*.png" % (i[aksara])))

    for y in range(1, count_image+1):
        abs_file_path = os.path.join(directory_path, "%s_%s.png" % (i[aksara],str(y)))
        img = Image.open(abs_file_path)
        img = np.array(img)
        img = img[:, :]
        final_np = np.append(final_np,img)
        final_label = np.append(final_label,aksara)
    sumImg = sumImg + count_image

    np.random.seed(42)

    if cfg.koropak28_test==False:
    	final_np = final_np.reshape((sumImg, 28, 28, 1)).astype(np.float32)
    	final_label = final_label.reshape((sumImg)).astype(np.int32)
    	
    	if sumImg < cfg.sampling_threshold:
    		idx = np.random.choice(sumImg,size=sumImg*2,replace=True)
    	else:
    		idx = np.random.choice(sumImg,size=sumImg-subtract,replace=False)
    	print("Idx = " + str(len(idx)))

    	final_np = final_np[idx,:]
    	final_label = final_label[idx]

    	return(final_np,final_label)

    else:

        directory_path = os.path.join(script_dir,"data","sunda_kuno","train-test_image","koropak_28")
        count_image = len(glob.glob1(directory_path,"%s_*.png" % (i[aksara])))
        print("Count_image koropak = " + str(count_image))

        for y in range(0, count_image):
            abs_file_path = os.path.join(directory_path, "%s_%s.png" % (i[aksara],str(y)))
            img = Image.open(abs_file_path)
            img = np.array(img)
            img = img[:, :]
            final_np = np.append(final_np,img)
            final_label = np.append(final_label,aksara)
        sumImg = sumImg + count_image

        directory_path = os.path.join(script_dir,"data","sunda_kuno","train-test_image","kawih_p")
        count_image = len(glob.glob1(directory_path,"%s_*.png" % (i[aksara])))
        print("Count_image kawih = " + str(count_image))

        for y in range(0, count_image):
            abs_file_path = os.path.join(directory_path, "%s_%s.png" % (i[aksara],str(y)))
            img = Image.open(abs_file_path)
            img = np.array(img)
            img = img[:, :]
            final_np = np.append(final_np,img)
            final_label = np.append(final_label,aksara)
        sumImg = sumImg + count_image

        print("SumImg = " + str(sumImg))

        final_np = final_np.reshape((sumImg, 28, 28, 1)).astype(np.float32)
        final_label = final_label.reshape((sumImg)).astype(np.int32)

        if cfg.sampling_threshold!=0:
            if sumImg < 50:
                idx = np.random.choice(sumImg,size=50,replace=True)
            else:
                idx = np.random.choice(sumImg,size=50,replace=False)
            print("Idx = " + str(len(idx)))
            final_np = final_np[idx,:]
            final_label = final_label[idx]
        else:
            final_np = final_np
            final_label = final_label

        # idx = np.random.choice(sumImg,size=50,replace=True)

        # if sumImg < cfg.sampling_threshold:
        #     idx = np.random.choice(sumImg,size=sumImg*2,replace=True)
        # else:
        #     idx = np.random.choice(sumImg,size=sumImg-subtract,replace=False)
        # print("Idx = " + str(len(idx)))

        # data_train, data_test, labels_train, labels_test = train_test_split(final_np[idx,:], final_label[idx], test_size=0.30, random_state=42)
        # # data_train, data_test, labels_train, labels_test = train_test_split(final_np, final_label, test_size=0.30, random_state=42)
        # if is_training == True:
        #     return(data_train,labels_train)
        # else:
        #     data_val, data_test, labels_val, labels_test = train_test_split(data_test, labels_test, test_size=0.50, random_state=42)
        #     return(data_val, data_test, labels_val, labels_test)

        # directory_path = os.path.join(script_dir,"data","sunda_kuno","train-test_image","koropak_28")
        # count_image = len(glob.glob1(directory_path,"%s_*.png" % (i[aksara])))

        # print("Count_image test = " + str(count_image))
        
        # for y in range(0, count_image):
        #     abs_file_path = os.path.join(directory_path, "%s_%s.png" % (i[aksara],str(y)))
        #     img = Image.open(abs_file_path)
        #     img = np.array(img)
        #     img = img[:, :]
        #     final_np = np.append(final_np,img)
        #     final_label = np.append(final_label,aksara)
        # sumImg = sumImg + count_image
        # print("SumImg = " + str(sumImg))
        # final_np = final_np.reshape((sumImg, 28, 28, 1)).astype(np.float32)
        # final_label = final_label.reshape((sumImg)).astype(np.int32)

        # if sumImg < cfg.sampling_threshold:
        # 	idx = np.random.choice(sumImg,size=sumImg*2,replace=True)
        # else:
        # 	idx = np.random.choice(sumImg,size=sumImg-subtract,replace=False)
        # print("Idx = " + str(len(idx)))

        # final_np = final_np[idx,:]
        # final_label = final_label[idx]

        return(final_np,final_label)

    #     data_train, data_test, labels_train, labels_test = train_test_split(final_np[idx,:], final_label[idx], test_size=0.30, random_state=42)
    #     if is_training == True:
    #     	return(data_train,labels_train)
    #     else:
    #     	data_val, data_test, labels_val, labels_test = train_test_split(data_test, labels_test, test_size=0.50, random_state=42)
    #     	return(data_val, data_test, labels_val, labels_test)

    # final_np = final_np.reshape((sumImg, 28, 28, 1)).astype(np.float32)
    # final_label = final_label.reshape((sumImg)).astype(np.int32)

    # idx = np.random.choice(sumImg,size=sumImg-subtract,replace=False)
    # final_np = final_np[idx,:]
    # final_label = final_label[idx]

    # return(final_np,final_label)

def get_khmer_each(is_training,aksara,subtract=0):
    script_dir = os.path.abspath('')
    input_class = [0,1,3,6,7,9,10,17,19,20,22,24,25,29,33,34,35,36,37,40,41,42,43,44,45,47,48,50]
    # i=["A","BA","CA","DA","GA","HA","I","JA","KA","LA","MA","NA","NGA","NYA","PA","PANELENG","PANEULEUNG","PANGHULU","PANGLAYAR","PANOLONG","PANYUKU","PATEN","RA","SA","TA","U","WA","YA"]
    
    final_np = np.array([])
    final_label = np.array([])

    sumImg = 0
    directory_path = os.path.join(script_dir,"data","khmer","train-test_image")
    count_image = len(glob.glob1(directory_path,"%s_*.png" % (input_class[aksara])))

    for y in range(0, count_image):
        abs_file_path = os.path.join(directory_path, "%s_%s.png" % (input_class[aksara],str(y)))
        img = Image.open(abs_file_path)
        img = np.array(img)
        img = img[:, :]
        final_np = np.append(final_np,img)
        final_label = np.append(final_label,aksara)
    sumImg = sumImg + count_image

    final_np = final_np.reshape((sumImg, 28, 28, 1)).astype(np.float32)
    final_label = final_label.reshape((sumImg)).astype(np.int32)

    idx = np.random.choice(sumImg,size=sumImg-subtract,replace=False)

    data_train, data_test, labels_train, labels_test = train_test_split(final_np[idx,:], final_label[idx], test_size=0.30, random_state=42)

    print("Data train = " + str(len(data_train)))

    if cfg.gan==True:
        sumImgTrain = len(data_train)
        
        final_np = np.array([])
        final_label = np.array([])

        sumImgGAN = 0
        directory_path = os.path.join(script_dir,"data","khmer","GAN")
        count_image = 5

        for y in range(51, 51+count_image):
            abs_file_path = os.path.join(directory_path, "%s_%s.png" % (input_class[aksara],str(y)))
            img = Image.open(abs_file_path)
            img = np.array(img)
            img = img[:, :]
            final_np = np.append(final_np,img)
            final_label = np.append(final_label,aksara)
        sumImgGAN = sumImgGAN + count_image
        sumImgTrain = sumImgTrain + sumImgGAN

        final_np = final_np.reshape((sumImgGAN, 28, 28, 1)).astype(np.float32)
        final_label = final_label.reshape((sumImgGAN)).astype(np.int32)

        print("Final_np = " + str(final_np.shape))
        print("Data train to be appended = " + str(data_train.shape))

        data_train = np.append(data_train,final_np)
        labels_train = np.append(labels_train,final_label)

        data_train = data_train.reshape((sumImgTrain, 28, 28, 1)).astype(np.float32)
        labels_train = labels_train.reshape((sumImgTrain)).astype(np.int32)

        print("Data train akhir = " + str(len(data_train)))

    if is_training == True:
        return(data_train,labels_train)
    else:
        data_val, data_test, labels_val, labels_test = train_test_split(data_test, labels_test, test_size=0.50, random_state=42)
        return(data_val, data_test, labels_val, labels_test)

def get_khmer_each_crossval(is_training,aksara,subtract=0):
    script_dir = os.path.abspath('')
    input_class = [0,1,3,6,7,9,10,17,19,20,22,24,25,29,33,34,35,36,37,40,41,42,43,44,45,47,48,50]
    # i=["A","BA","CA","DA","GA","HA","I","JA","KA","LA","MA","NA","NGA","NYA","PA","PANELENG","PANEULEUNG","PANGHULU","PANGLAYAR","PANOLONG","PANYUKU","PATEN","RA","SA","TA","U","WA","YA"]
    
    final_np = np.array([])
    final_label = np.array([])

    sumImg = 0
    directory_path = os.path.join(script_dir,"data","khmer","train-test_image")
    count_image = len(glob.glob1(directory_path,"%s_*.png" % (input_class[aksara])))

    for y in range(0, count_image):
        abs_file_path = os.path.join(directory_path, "%s_%s.png" % (input_class[aksara],str(y)))
        img = Image.open(abs_file_path)
        img = np.array(img)
        img = img[:, :]
        final_np = np.append(final_np,img)
        final_label = np.append(final_label,aksara)
    sumImg = sumImg + count_image
    final_np = final_np.reshape((sumImg, 28, 28, 1)).astype(np.float32)
    final_label = final_label.reshape((sumImg)).astype(np.int32)

    idx = np.random.choice(sumImg,size=sumImg-subtract,replace=False)
    final_np = final_np[idx,:]
    final_label = final_label[idx]
    return(final_np,final_label)

def get_khmer(is_training):
    input_class = [0,1,3,6,7,9,10,17,19,20,22,24,25,29,33,34,35,36,37,40,41,42,43,44,45,47,48,50]

    script_dir = os.path.dirname(__file__)
    # directory_path = os.path.join(script_dir,"train-test_image")
    # count_image = len(glob.glob1(directory_path,"%s_*.png" % (i[aksara])))

    sumImg = 0
    final_np = np.array([])
    final_label = np.array([])

    if is_training==True:
        for x in range(0, len(input_class), 1):
            final_np_tmp, final_label_tmp = get_khmer_each(True,x,0)
            
            final_np = np.append(final_np,final_np_tmp)
            final_label = np.append(final_label,final_label_tmp)
            
            sumImg = sumImg + len(final_np_tmp)
            
        sumImgTrain = sumImg
        print(final_label.shape)

        for x in range(0, len(input_class), 1):
            final_np_tmp, _ , final_label_tmp , _ = get_khmer_each(False,x,0)
            
            final_np = np.append(final_np,final_np_tmp)
            final_label = np.append(final_label,final_label_tmp)

            sumImg = sumImg + len(final_np_tmp)
            
        sumImgVal = sumImg - sumImgTrain
        print(sumImg)
        print(final_label.shape)

        final_np = final_np.reshape((sumImg, 28, 28, 1)).astype(np.float32)
        final_label = final_label.reshape((sumImg)).astype(np.int32)
        print(final_np.shape)
        return(final_np,final_label,sumImgTrain,sumImgVal)
        
    else:
        for x in range(0, len(input_class), 1):
            _, final_np_tmp, _, final_label_tmp = get_khmer_each(False,x,0)
            
            final_np = np.append(final_np,final_np_tmp)
            final_label = np.append(final_label,final_label_tmp)

            sumImg = sumImg + len(final_np_tmp)
            
        final_np = final_np.reshape((sumImg, 28, 28, 1)).astype(np.float32)
        final_label = final_label.reshape((sumImg)).astype(np.int32)
        return(final_np,final_label,sumImg)

def get_khmer_crossval(is_training):
    input_class = [0,1,3,6,7,9,10,17,19,20,22,24,25,29,33,34,35,36,37,40,41,42,43,44,45,47,48,50]
    script_dir = os.path.dirname(__file__)
    # directory_path = os.path.join(script_dir,"train-test_image")

    sumImg = 0
    final_np = np.array([])
    final_label = np.array([])

    for x in range(0, len(input_class), 1):
        final_np_tmp, final_label_tmp = get_khmer_each_crossval(True,x,0)
        
        final_np = np.append(final_np,final_np_tmp)
        final_label = np.append(final_label,final_label_tmp)
        
        sumImg = sumImg + len(final_np_tmp)
    final_np = final_np.reshape((sumImg, 28, 28, 1)).astype(np.float32)
    final_label = final_label.reshape((sumImg)).astype(np.int32)

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold=1
    for train_index, test_index in kf.split(final_np, final_label):
        print("Fold = " + str(fold))
        if cfg.k_fold == fold:
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = final_np[train_index], final_np[test_index]
            y_train, y_test = final_label[train_index], final_label[test_index]
            print(y_test)
            if is_training==True:
                print("Training...")
                data_train, data_val, labels_train, labels_val = train_test_split(X_train, y_train, stratify = y_train, test_size=0.25, random_state=42)
                return(data_train, data_val, labels_train, labels_val)
            else:
                print("Testing...")
                return(X_test,y_test,len(X_test))

        fold=fold+1

def get_sunda_kuno_crossval(is_training):
    i=["A","BA","CA","DA","GA","HA","I","JA","KA","LA","MA","NA","NGA","NYA","PA","PANELENG","PANEULEUNG","PANGHULU","PANGLAYAR","PANOLONG","PANYUKU","PATEN","RA","SA","TA","U","WA","YA"]
    # l=[0,0,0,22,0,0,0,0,14,3,0,84,0,0,14,0,17,14,0,0,0,28,8,49,35,0,0,0]
    script_dir = os.path.dirname(__file__)
    directory_path = os.path.join(script_dir,"data","sunda_kuno","train-test_image",str(cfg.dataset_pre))

    sumImg = 0
    final_np = np.array([])
    final_label = np.array([])

    for x in range(0, len(i), 1):
        final_np_tmp, final_label_tmp = get_sunda_kuno_each_crossval(True,x,0)
        
        final_np = np.append(final_np,final_np_tmp)
        final_label = np.append(final_label,final_label_tmp)
        
        sumImg = sumImg + len(final_np_tmp)
    final_np = final_np.reshape((sumImg, 28, 28, 1)).astype(np.float32)
    final_label = final_label.reshape((sumImg)).astype(np.int32)

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold=1
    for train_index, test_index in kf.split(final_np, final_label):
        print("Fold = " + str(fold))
        if cfg.k_fold == fold:
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = final_np[train_index], final_np[test_index]
            y_train, y_test = final_label[train_index], final_label[test_index]
            print(y_test)
            if is_training==True:
                print("Training...")
                data_train, data_val, labels_train, labels_val = train_test_split(X_train, y_train, stratify = y_train, test_size=0.25, random_state=42)
                return(data_train, data_val, labels_train, labels_val)
            else:
                print("Testing...")
                return(X_test,y_test,len(X_test))

        fold=fold+1

def get_sunda_kuno(is_training):
    i=["A","BA","CA","DA","GA","HA","I","JA","KA","LA","MA","NA","NGA","NYA","PA","PANELENG","PANEULEUNG","PANGHULU","PANGLAYAR","PANOLONG","PANYUKU","PATEN","RA","SA","TA","U","WA","YA"]
    # i=["A","BA","CA","DA","GA","HA","I","JA","KA","LA","MA","NA","NGA","NYA","PA","RA","SA","TA","U","WA","YA"]
    l=[0,0,0,22,0,0,0,0,14,3,0,84,0,0,14,0,17,14,0,0,0,28,8,49,35,0,0,0]
    minmax_num = [88,107,77,142,102,92,70,86,140,116,131,210,88,72,145,108,113,186,81,95,155,107,134,155,165,72,95,80]
    # j=[30,47,19,67,37,27,16,21,60,60,56,120,25,14,61,56,90,78,18,24,22]
    # k=[12,30,7,45,16,11,7,9,40,25,24,80,10,6,39,36,60,52,7,10,9]
    script_dir = os.path.dirname(__file__)
    # directory_path = os.path.join(script_dir,"train-test_image")
    # count_image = len(glob.glob1(directory_path,"%s_*.png" % (i[aksara])))

    sumImg = 0
    final_np = np.array([])
    final_label = np.array([])

    if is_training==True:
        for x in range(0, len(i), 1):
            # final_np_tmp, final_label_tmp = get_sunda_kuno_each(True,x,l[x])
            final_np_tmp, final_label_tmp = get_sunda_kuno_each(True,x,0)
            
            if len(final_np_tmp) < minmax_num[x]:
                idx = np.random.choice(len(final_np_tmp),size=minmax_num[x],replace=True)
            else:
                idx = np.random.choice(len(final_np_tmp),size=minmax_num[x],replace=False)

            final_np = np.append(final_np,final_np_tmp)
            final_label = np.append(final_label,final_label_tmp)
            
            sumImg = sumImg + len(final_np_tmp)

            # if cfg.gan==True:
            #     if sumImg<minmax_num[x]:
            #         gan = GAN()
            #         final_np_tmp, final_label_tmp = gan.train(aksara=x, aksara_num=minmax_num[x]-sumImg ,epochs=30001, batch_size=32, sample_interval=5000)
            #         print("GAN: +" + str(len(final_np_tmp)))
            #         final_np = np.append(final_np,final_np_tmp)
            #         final_label = np.append(final_label,final_label_tmp)
            #         sumImg = len(final_np_tmp)

            #     idx = np.random.choice(sumImg,size=minmax_num[x],replace=False)
            #     print("Idx = " + str(len(idx)))
            #     final_np = final_np[idx,:]
            #     final_label = final_label[idx]
            #     sumImg = len(final_np)

#             directory_path = os.path.join(script_dir,"data","sunda_kuno","ready_to_train_3")
#             # directory_path = os.path.join(script_dir,"data","sunda_kuno","ready_to_train_2")
#             # directory_path = os.path.join(script_dir,"data","sunda_kuno","ready_to_train")
#             countImg = len(glob.glob1(directory_path,"%s_*.png" % (i[x])))
#             sumImg = sumImg + countImg
#             for y in range(1, countImg+1):
#                 rel_path = "%s_%s.png" % (i[x],str(y))
#                 abs_file_path = os.path.join(directory_path, rel_path)
#                 img = Image.open(abs_file_path)
#                 img = np.array(img)
# #                 print(img.shape)/
#                 img = img[:, :]
#                 final_np = np.append(final_np,img)
#                 final_label = np.append(final_label,x)

        sumImgTrain = sumImg
        print(final_label.shape)

        for x in range(0, len(i), 1):
            # final_np_tmp, _ , final_label_tmp , _ = get_sunda_kuno_each(False,x,l[x])
            final_np_tmp, _ , final_label_tmp , _ = get_sunda_kuno_each(False,x,0)
            
            final_np = np.append(final_np,final_np_tmp)
            final_label = np.append(final_label,final_label_tmp)

            sumImg = sumImg + len(final_np_tmp)
            
            # directory_path = os.path.join(script_dir,"data","sunda_kuno","ready_to_test_2")
            # countImg = len(glob.glob1(directory_path,"%s_*.png" % (i[x])))
            # sumImg = sumImg + int(k[x]/2)
            # for y in range(1, int((k[x])/2)+1):
            #     rel_path = "%s_%s.png" % (i[x],str(y))
            #     abs_file_path = os.path.join(directory_path, rel_path)

            #     img = Image.open(abs_file_path)
            #     img = np.array(img)
            #     img = img[:, :]
            #     final_np = np.append(final_np,img)
            #     final_label = np.append(final_label,x)
        sumImgVal = sumImg - sumImgTrain
        print(sumImg)
        print(final_label.shape)

        final_np = final_np.reshape((sumImg, 28, 28, 1)).astype(np.float32)
        final_label = final_label.reshape((sumImg)).astype(np.int32)
        print(final_np.shape)
        return(final_np,final_label,sumImgTrain,sumImgVal)
        
    else:
        for x in range(0, len(i), 1):
            _, final_np_tmp, _, final_label_tmp = get_sunda_kuno_each(False,x,l[x])
            
            final_np = np.append(final_np,final_np_tmp)
            final_label = np.append(final_label,final_label_tmp)

            sumImg = sumImg + len(final_np_tmp)
            
            # sumImg = sumImg - (-k[x]//2)
            # for y in range((-(-k[x])//2)+1, k[x]+1):
            #     rel_path = "%s_%s.png" % (i[x],str(y))
            #     abs_file_path = os.path.join(script_dir,"data","sunda_kuno","ready_to_test_2", rel_path)
            #     # abs_file_path = os.path.join(script_dir,"data","sunda_kuno","ready_to_test", rel_path)

            #     img = Image.open(abs_file_path)
            #     img = np.array(img)
            #     img = img[:, :]
            #     final_np = np.append(final_np,img)
            #     final_label = np.append(final_label,x)
        final_np = final_np.reshape((sumImg, 28, 28, 1)).astype(np.float32)
        final_label = final_label.reshape((sumImg)).astype(np.int32)
        return(final_np,final_label,sumImg)

def load_data(dataset, batch_size, is_training=True, one_hot=False):
    print("Dataset: " + dataset)
    if dataset == 'mnist':
        return load_mnist(batch_size, is_training)
    elif dataset == 'fashion-mnist':
        return load_fashion_mnist(batch_size, is_training)
    elif dataset == 'sunda_kuno':
        return load_sunda_kuno(batch_size, is_training)
    elif dataset == 'khmer':
        return load_khmer(batch_size, is_training)
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dataset)


def get_batch_data(dataset, batch_size, num_threads):
    if dataset == 'mnist':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_mnist(batch_size, is_training=True)
    elif dataset == 'fashion-mnist':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_fashion_mnist(batch_size, is_training=True)
    elif dataset == 'sunda_kuno':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_sunda_kuno(batch_size, is_training=True)
    elif dataset == 'khmer':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_khmer(batch_size, is_training=True)
    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=num_threads,
                                  batch_size=batch_size,
                                  capacity=batch_size * 64,
                                  min_after_dequeue=batch_size * 32,
                                  allow_smaller_final_batch=False)

    return(X, Y)


def save_images(imgs, size, path):
    '''
    Args:
        imgs: [batch_size, image_height, image_width]
        size: a list with tow int elements, [image_height, image_width]
        path: the path to save images
    '''
    imgs = (imgs + 1.) / 2  # inverse_transform
    return(scipy.misc.imsave(path, mergeImgs(imgs, size)))


def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs


# For version compatibility
def reduce_sum(input_tensor, axis=None, keepdims=False):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims)


# For version compatibility
def softmax(logits, axis=None):
    try:
        return tf.nn.softmax(logits, axis=axis)
    except:
        return tf.nn.softmax(logits, dim=axis)


def get_shape(inputs, name=None):
    name = "shape" if name is None else name
    with tf.name_scope(name):
        static_shape = inputs.get_shape().as_list()
        dynamic_shape = tf.shape(inputs)
        shape = []
        for i, dim in enumerate(static_shape):
            dim = dim if dim is not None else dynamic_shape[i]
            shape.append(dim)
        return(shape)

def resize(file_name, invert):
    img = Image.open(file_name).convert('LA')
    enhancer = ImageEnhance.Contrast(img)
    
    factor = 2.0
    img = enhancer.enhance(factor)
    img = img.filter(ImageFilter.SMOOTH)
    img = img.filter(ImageFilter.SHARPEN)
    img = img.resize((28, 28))
    
    img.save("temp.png")
    
    img = cv.imread('temp.png', 0)
    
    if invert==True:
        img = cv.bitwise_not(img)
    ret,th = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    
    blur = cv.GaussianBlur(img,(5,5),0)
    ret2,th2 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    
    mask = np.where(th ==  0, th2, 255)
    img = np.where(mask == 0, img, 255)
    img = cv.bitwise_not(img)
    img = Image.fromarray(img)
  
    return img

class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, aksara, aksara_num, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        if cfg.k_fold==0:
            X_train, y_train = get_sunda_kuno_each(True,aksara,subtract=0)
        else:
            X_train, y_train = get_sunda_kuno_each_crossval(True,aksara,subtract=0)
        # X_train, sumTrain = get_sunda(aksara)
        sumTrain = len(X_train)
        X_train = X_train[:,:,:,0]

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        min_loss = 100

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            if epoch == epochs-1:
                return self.sample_images(epoch, aksara, sumTrain, aksara_num)

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                print ("%d-%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (aksara, epoch, d_loss[0], 100*d_loss[1], g_loss))
                # self.sample_images(epoch, aksara, sumTrain, aksara_num)
            
#             if d_loss[0] + g_loss < min_loss and epoch > 10000:
#                 min_loss = d_loss[0] + g_loss
#                 self.sample_images(epoch, aksara, sumTrain, aksara_num)

    def sample_images(self, epoch, aksara, sumTrain, aksara_num):
        aksaraList=["A","BA","CA","DA","GA","HA","I","JA","KA","LA","MA","NA","NGA","NYA","PA","PANELENG","PANEULEUNG","PANGHULU","PANGLAYAR","PANOLONG","PANYUKU","PATEN","RA","SA","TA","U","WA","YA"]
        r, c = 10, 10
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        gan_train_directory_path = "data/sunda_kunno/kawih_GAN/"

        final_np = np.array([])
        final_label = np.array([])
        
        for i in range(0,aksara_num):
            img = gen_imgs[i,:,:,0]
            normalizedImg = np.zeros((28,28))
            normalizedImg = cv.normalize(img, normalizedImg, 0, 255, cv.NORM_MINMAX)

            file_name = "%s_%s.png" % (aksaraList[aksara],str(sumTrain+i+1))
            cv.imwrite(gan_train_directory_path + "raw/" + file_name, normalizedImg)
            
            img = resize(gan_train_directory_path + "raw/" + file_name, invert=False)
            img.save(gan_train_directory_path + "resized/" + file_name)

            img = Image.open(gan_train_directory_path + "resized/" + file_name)
            img = np.array(img)
            img = img[:, :]
            final_np = np.append(final_np,img)
            final_label = np.append(final_label,aksara)

        return final_np, final_label
            
        # for i in range(0,10):
        #     img = gen_imgs[i,:,:,0]
        #     normalizedImg = np.zeros((28,28))
        #     normalizedImg = cv.normalize(img, normalizedImg, 0, 255, cv.NORM_MINMAX)
        #     cv.imwrite("GAN_generated_images/%s_Cadangan_%s.png" % (aksaraList[aksara],str(i+1)), normalizedImg)
