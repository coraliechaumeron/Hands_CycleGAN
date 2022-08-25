#!/usr/bin/env python
# coding: utf-8

# In[104]:

import matplotlib
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import *
from tensorflow_addons.layers import *
from PIL import Image
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import os

import matplotlib.pyplot

from skimage.color import rgb2gray
from skimage import color

from tensorflow.keras.utils import plot_model

import os as os
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

from random import random
from numpy import load
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from matplotlib import pyplot

from tensorflow.keras import Input

from random import random
from numpy import load
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randint

from os import listdir
from numpy import asarray
from numpy import vstack
from numpy import savez_compressed

from tensorflow import keras
import tensorflow.keras.preprocessing.image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img







# Chargement des images pour création d'un dataset à partir d'un dossier 
# qui contient les images (dans 2 dossiers tests et train eux-mêmes composés de 2 dossiers trainA et trainB puis testA et testB
# Puis prétraitement des images en les mettant au bon format et en les normalisant 

path_train = 'mettre le chamin qui emmène dans le dossier train et qui initie l ouverture des deux sous dossiers en écrivant just train : ex :/home.local/chaumeron/essai_cycle_GAN/Image/separate3/train/train '
path_test = 'mettre le chamin qui emmène dans le dossier train et qui initie l ouverture des deux sous dossiers en écrivant just train : ex :/home.local/chaumeron/essai_cycle_GAN/Image/separate3/train/tests '
path_save = 'chemin du dossier dans lequel on souhaire enregistrer les résultats au fur et à mesure et les courbes du loss'

def preprocess(records):
    images =  records['image']
    images = tf.cast(images, tf.float32)/255.0
    return images

def tf_pipeline(dataset):
    dataset = tf.data.Dataset.from_tensor_slices({'image':dataset})
    dataset = dataset.map(preprocess)
    dataset = dataset.repeat().shuffle(100).batch(16).prefetch(1)
    return dataset
    
def tf_data(path):
    trainingA = []
    for x in glob.glob(path+'A/*'):
        image = Image.open(x)#.convert('LA')
        image = image.resize((128,128))
        trainingA.append(np.array(image))  
    trainingB = []
    for x in glob.glob(path+'B/*'):
        image = Image.open(x)#.convert('LA')
        image = image.resize((128,128))
        trainingB.append(np.array(image))
    a,b = tf_pipeline(trainingA),tf_pipeline(trainingB)
    return a.__iter__(),b.__iter__()

def tf_pipeline_aff(dataset):
    dataset = tf.data.Dataset.from_tensor_slices({'image':dataset})
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(16)#.prefetch(1)
    return dataset

def tf_data_aff(path):
    trainingA = []
    for x in glob.glob(path+'A/*'):
        image = Image.open(x)#.convert('LA')
        image = image.resize((128,128))
        trainingA.append(np.array(image))  
    trainingB = []
    for x in glob.glob(path+'B/*'):
        image = Image.open(x)#.convert('LA')
        image = image.resize((128,128))
        trainingB.append(np.array(image))
    a,b = tf_pipeline_aff(trainingA),tf_pipeline_aff(trainingB)
    return a.__iter__(),b.__iter__()

trainA,trainB = tf_data(path)





# Valeurs des paramètres à modifier et paramètres d'entrée en généal

input_dim = (128,128,3)      # input/output image dimension
depth = 5                    # network depth  
kernel = 3                   # kernel size for Conv2D
n_batch = 16                 # batch_size
epochs = 150
resnet = 8		     # Nombre de blocs résiduels dans le cas d'un générateur avec des blocs résiduels
steps = round(400/n_batch)   #steps per epoch, we have ~1500 samples per domain so calculating steps using it


# Construction des discriminateurs

def define_discriminator(image_shape):
	# weight initialization : distribution aléatoire
	init = RandomNormal(stddev=0.02)
	# source image input
	in_image = tensorflow.keras.Input(shape=image_shape)
	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)  # padding : same ou valid : changement du noyau de convolution
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)  # ne faut-il pas mettre 3 si on veut du RGB : veut-on réellement du RGB
	# define model
	model = Model(in_image, patch_out)
	# compile model
	model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
	return model

image_shape=(128,128,3)
DiscA=define_discriminator(image_shape)
DiscB=define_discriminator(image_shape)
DiscA.summary()

# Affichage des couches qui nécessite le package graphivz 
#plot_model(DiscA, #to_file = "model_plot.png",show_shapes = True, show_layer_names = True) 


# Construction des générateurs 

## Cas 1 : s'il est composé de blocs résiduels

def resnet_block(n_filters, input_layer):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# first layer convolutional layer
	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# second convolutional layer
	g = Conv2D(n_filters, (1,1), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	# concatenate merge channel-wise with input layer
	g = Concatenate()([g, input_layer])
	return g


## Cas 2 : s'il s'agit d'un U-net

def conv_block(input_img, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input_img)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(input_img, num_filters):
    x = conv_block(input_img, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(input_img, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input_img)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(image_shape):
    in_image = tensorflow.keras.Input(shape=image_shape) 

    s1, p1 = encoder_block(in_image, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    out_image = Conv2D(3, 1, padding="same", activation="tanh")(d4)

    model = Model(in_image, out_image )
    return model 


# Cas 3 : s'il sagit de couches simples successives 

def define_generator(image_shape, n_resnet):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = tensorflow.keras.Input(shape=image_shape)  #prends une image aux dimensions voulues ici image_shape

	g = Conv2D(64, (5,5), padding='same', kernel_initializer=init)(in_image)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d128
	g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d256
	g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# R256
	for i in range(n_resnet):
		g = resnet_block(256, g)
	# u128
	g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# u64
	g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)

	g = Conv2D(3, (5,5), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	out_image = Activation('tanh')(g)
	# define model
	model = Model(in_image, out_image)
	return model


# Défiinit les 2 générateurs qui seront utilisés : ici le cas n°3
genA=define_generator(image_shape, resnet)
genB=define_generator(image_shape, resnet)
genA.summary()

#plot_model (genA, to_file = "model_genA.png",show_shapes = True, show_layer_names = True)


# Construction du modèle combiné (demi-GAN)

def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
	# ensure the model we're updating is trainable
	g_model_1.trainable = True
	# mark discriminator as not trainable
	d_model.trainable = False
	# mark other generator model as not trainable
	g_model_2.trainable = False
	# discriminator element
	input_gen = tensorflow.keras.Input(shape=image_shape)
	gen1_out = g_model_1(input_gen)
	output_d = d_model(gen1_out)
	# identity element
	input_id = tensorflow.keras.Input(shape=image_shape)
	output_id = g_model_1(input_id)
	# forward cycle
	output_f = g_model_2(gen1_out)
	# backward cycle
	gen2_out = g_model_2(input_id)
	output_b = g_model_1(gen2_out)
	# define model graph
	model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
	# define optimization algorithm configuration
	opt = Adam(lr=0.0002, beta_1=0.5)
	# compile model with weighting of least squares loss and L1 loss
	model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
	return model

comb_modelA=define_composite_model(genA,DiscA,genB,image_shape)
comb_modelB=define_composite_model(genB,DiscB,genA,image_shape)

#plot_model (comb_modelA, #to_file = "model_plot.png",show_shapes = True, show_layer_names = True)


# The generate_real_samples() function below implements this

# select a batch of random samples, returns images and target

def generate_real(dataset, batch_size,patch_size):
    labels = np.ones((batch_size,patch_size,patch_size,1))
    return dataset,labels
def generate_fake(dataset,g,batch_size, patch_size):
    predicted = g(dataset)
    labels = np.zeros((batch_size,patch_size,patch_size,1))
    return predicted,labels


# Enregistre les checkpoints et les écrase au fur et à mesure pour enregistrer ceeux des trois derniers steps

checkpoint_dir = './cyclegan'

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(genB=genB, genA=genA,DiscA=DiscA,DiscB=DiscB,comb_modelB=comb_modelB, comb_modelA=comb_modelA)
manager = tf.train.CheckpointManager(checkpoint, 'training_checkpoints13', max_to_keep=3)
checkpoint.restore(manager.latest_checkpoint)




# Entrainement

def train(discriminator_A, discriminator_B, generator_A_B, generator_B_A, composite_A_B, composite_B_A, epochs, batch_size, steps,n_patch):
    
    x=[]
    y=[]
    b=[]
    i=0
    m=50
    n=100


    for epoch in range(1,epochs):
        print(epoch)
        for step in range(1,steps):
            #print(epoch,step)
            
            x.append(i)

            x_real_A, y_real_A = generate_real(next(trainA),n_batch,n_patch)
            x_real_B, y_real_B = generate_real(next(trainB),n_batch,n_patch)

            
            x_fake_A, y_fake_A = generate_fake(x_real_B, generator_B_A,batch_size, n_patch)
            x_fake_B, y_fake_B = generate_fake(x_real_A, generator_A_B,batch_size, n_patch)
            
            g_A_B_loss,_,_,_,_ = composite_A_B.train_on_batch([x_real_A,x_real_B],[y_real_B,x_real_B, x_real_A, x_real_B])
            disc_A_real_loss = discriminator_A.train_on_batch(x_real_A, y_real_A)
            disc_A_fake_loss = discriminator_A.train_on_batch(x_fake_A, y_fake_A)
            
            g_B_A_loss,_,_,_,_ = composite_B_A.train_on_batch([x_real_B,x_real_A],[y_real_A,x_real_A, x_real_B, x_real_A])
            disc_B_real_loss = discriminator_B.train_on_batch(x_real_B, y_real_B)
            disc_B_fake_loss = discriminator_B.train_on_batch(x_fake_B, y_fake_B)
            
           # print('g_A_B_loss',g_A_B_loss)
           # print('g_B_A_loss',g_B_A_loss)
            
            y.append(g_A_B_loss)
            b.append(disc_A_real_loss)
            manager.save()

            i=i+1
        
# Enregistrement tous les 10 epochs des images générées (à patir du dossier test)
	
        if epoch%10==0:
            testA,testB = tf_data('path_test')
            x_real_A, _ = generate_real(next(testA),n_batch,0)
            images_B,_ = generate_fake(x_real_A, genB,n_batch,0)
            images_C, _ = generate_fake(images_B, genA,n_batch,0)



            fig,ax = plt.subplots(n_batch,figsize=(75,75))
            for index,img in enumerate(zip(x_real_A,images_B, images_C)):
                concat_numpy = np.clip(np.hstack((img[0],img[1], img[2])),0,1)
                ax[index].imshow((concat_numpy* 255).astype(np.uint8))
                plt.imshow((concat_numpy * 255).astype(np.uint8))
                filename1 = 'generated_plot_epochs'+str(epoch+80)+'_aff_%03d.png' % (index+1)
                plt.savefig('path_save' + filename1)
                plt.close()
            fig.tight_layout()
            
        else : 
            pass
            
	
# Affichage des courbes de losss (2/4) : 	
	
    plt.plot(x,y,b)
    filename2 = 'loss_GAB+disc_A_real_loss_250'
    plt.savefig('/home.local/chaumeron/essai_cycle_GAN/tests/better/results/resnet/13/' + filename2)
    #plt.show()

    plt.close()

    
    
    

train(DiscA, DiscB, genB, genA, comb_modelB, comb_modelA, epochs, n_batch, steps, DiscA.output_shape[1])    


# test final sur les images du dossier test à partir du dernier checkpoint (affichage image et loss)

testA,testB = tf_data_aff('path_test')

x_real_A, _ = generate_real(next(testA),n_batch,0)
images_B,_ = generate_fake(x_real_A, genB,n_batch,0)
images_C, _ = generate_fake(images_B, genA,n_batch,0)



fig,ax = plt.subplots(n_batch,figsize=(75,75))
for index,img in enumerate(zip(x_real_A,images_B, images_C)):
    concat_numpy = np.clip(np.hstack((img[0],img[1], img[2])),0,1)
    ax[index].imshow((concat_numpy* 255).astype(np.uint8))
    plt.imshow((concat_numpy * 255).astype(np.uint8))
    filename1 = 'generated_plot_epochs250_aff%03d.png' % (index+1)
    plt.savefig('path_save' + filename1)
    plt.close()
fig.tight_layout()




# In[16]:

x_real_B,_ = generate_real(next(testB),n_batch,0)
images_A,_ = generate_fake(x_real_B, genA,n_batch, 0)
fig,ax = plt.subplots(n_batch,figsize=(75,75))
for index,img in enumerate(zip(x_real_B,images_A)):
    concat_numpy = np.hstack((img[0],img[1]))
    ax[index].imshow(concat_numpy)
fig.tight_layout()

