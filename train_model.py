import numpy as np 
import pandas as pd
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfl
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split

#Load data
data = pd.read_csv("/kaggle/input/airbus-ship-detection/train_ship_segmentations_v2.csv")

#Here we can see that it images were encoded by rle and NaN rows mean that's image without ship
#Adding new column which can show is ship in image
data['has_sheep']=data['EncodedPixels'].notna().astype(int)

#Let's count and compare images
#Ass we see images without ship more than it present, that can do disbalance in our model
ship_count = data['has_sheep'].value_counts()
ship_count.plot(kind='bar')
plt.show()
ship_count

#So, we need to cut some data to improve accuracy in our model
#It will be 9000 pictures with ships and 1000 without
empty, has_ships = 1000, 9000
df = pd.concat([data[data["EncodedPixels"].isna()].sample(empty), data[~data["EncodedPixels"].isna()].sample(has_ships)])
df.shape

SIZE_FULL = 768 # original dataset's images size
SIZE = 256 #size used for using in model (3x3 squares to make original image)
IMG_CHANNELS = 3 # image channels

def crop3x3(img, i):
    """img: np.ndarray - original image 768x768
       i: int 0-8 - image index from crop: 0 1 2
                                           3 4 5
                                           6 7 8
       returns: image 256x256 
    """
    return img[(i//3)*SIZE: ((i//3)+1)*SIZE,(i%3)*SIZE: (i%3+1)*SIZE]


def crop3x3_mask(img):
    """Returns crop image, crop index with maximum ships area"""
    i = K.argmax((
        K.sum(crop3x3(img, 0)),
        K.sum(crop3x3(img, 1)),
        K.sum(crop3x3(img, 2)),
        K.sum(crop3x3(img, 3)),
        K.sum(crop3x3(img, 4)),
        K.sum(crop3x3(img, 5)),
        K.sum(crop3x3(img, 6)),
        K.sum(crop3x3(img, 7)),
        K.sum(crop3x3(img, 8)),
    ))
    return (crop3x3(img, i), i)

# https://www.kaggle.com/code/stainsby/fast-tested-rle/notebook
def decode(mask_rle):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    img=np.zeros(SIZE_FULL*SIZE_FULL, dtype=np.float32)
    if not(type(mask_rle) is float):
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1.0
    return img.reshape((SIZE_FULL, SIZE_FULL)).T

class TrainDataGenerator(tfk.utils.Sequence):
    '''
    This class allows to efficiently prepare and receive batches of data for model training, 
    which is important when working with large amounts of data, such as training image sets
    '''
    
    def __init__(self, datapath ,batch_size, df_mask: pd.DataFrame):
        self.datapath = datapath
        self.batch_size = batch_size
        self.df =  df_mask.sample(frac=1)
        self.l = len(self.df)//batch_size

    def __len__(self):
        return self.l

    def on_epoch_end(self):
        pass

    def __getitem__(self, index):
        mask = np.empty((self.batch_size, SIZE , SIZE), np.float32)
        image = np.empty((self.batch_size, SIZE, SIZE, 3), np.float32)
        
        for b in range(self.batch_size):
            temp = tfk.preprocessing.image.load_img(self.datapath + '/' + self.df.iloc[index*self.batch_size+b]['ImageId'])
            temp = tfk.preprocessing.image.img_to_array(temp)/255
        
            mask[b], i = crop3x3_mask( # decoding mask from run-length format, and cropping part with maximum ship's area(№ i)
                decode(
                    self.df.iloc[index*self.batch_size+b]['EncodedPixels']
                )
            ) 
            image[b] = crop3x3(temp, i) # using corresponding to mask crop of image (№ i)
            
        return image, mask

def dice_score(y_true, y_pred):
    '''calculating dice score, will used for evaluate model'''
    return (2.0*K.sum(y_pred * y_true)+0.0001) / (K.sum(y_true)+ K.sum(y_pred)+0.0001)

def BFCE_dice(y_true, y_pred):
    return  K.binary_focal_crossentropy(y_true, y_pred)+  (1-dice_score(y_true, y_pred))*0.1

def BCE_dice(y_true, y_pred):
    return  K.binary_crossentropy(y_true, y_pred)+  (1-dice_score(y_true, y_pred))

# dropout = 0.2
dropout = 0.2

k =2
def dconv(prev, filters, kernel_size=3):
    prev = tfl.BatchNormalization()(prev)
    prev = tfl.Conv2D(filters, kernel_size, padding="same", activation="elu", kernel_initializer= 'he_normal')(prev)
    prev = tfl.Dropout(dropout)(prev)
    prev = tfl.Conv2D(filters, kernel_size, padding="same", activation="elu", kernel_initializer= 'he_normal')(prev)
    return prev
    


def down(prev, filters, kernel_size=3): 
    skip = dconv(prev, filters, kernel_size)
    prev = tfl.MaxPool2D(strides=2, padding='valid')(skip)
    return prev


def bridge(prev, filters,kernel_size=3):  
    prev = dconv(prev, filters, kernel_size)
    prev = tfl.Conv2DTranspose(filters // 2, 2, strides=(2, 2))(prev)
    return prev


def up(prev, skip, filters, kernel_size=3):  
    prev = tfl.concatenate([prev, skip], axis=3) 
    prev = tfl.Dropout(dropout)(prev)
    prev = dconv(prev, filters, kernel_size)
    prev = tfl.Conv2DTranspose(filters // 2, 2, strides=(2, 2))(prev)
    return prev


def last(prev, skip, filters,kernels_size=(3,3)):
    prev = tfl.concatenate([prev, skip], axis=3)
    prev = tfl.Dropout(dropout)(prev)
    prev = tfl.Conv2D(filters, kernels_size[0], padding="same",activation="elu", kernel_initializer= 'he_normal')(prev)
    prev = tfl.Conv2D(filters, kernels_size[1], padding="same",activation="elu", kernel_initializer= 'he_normal')(prev)
    prev = tfl.Conv2D(filters=1, kernel_size=1,padding="same", activation="sigmoid")(prev)
    return prev


def unet_model(input_shape):
    inp = tfk.Input(shape=input_shape)
    inp = tfl.BatchNormalization()(inp)
    out, skip_1 = down(inp, k*16)
    out, skip_2 = down(out, k*32)
    out, skip_3 = down(out, k*64)
    out, skip_4 = down(out, k*128)
    out = bridge(out, k*256)
    out = up(out, skip_4, k*128)
    out = up(out, skip_3, k*64)
    out = up(out, skip_2, k*32)
    out = last(out, skip_1, k*16)

    model = tfk.Model(inputs=inp, outputs=out)
    return model

batch_size = 16
X_train, Y_train = train_test_split(df, test_size=0.2)
X_train = TrainDataGenerator("/kaggle/input/airbus-ship-detection/train_v2", batch_size, X_train)
Y_train = TrainDataGenerator("/kaggle/input/airbus-ship-detection/train_v2", batch_size, Y_train)
model = unet_model((SIZE, SIZE, 3))

model.compile(tf.keras.optimizers.Adam(0.0001) , BCE_dice  , dice_score)

callback = tfk.callbacks.ModelCheckpoint("./models/model.{epoch:02d}-{val_loss:.4f}-dice:{val_dice_score:.4f}.h5", "val_loss", save_best_only=True, save_weights_only=True)

history = model.fit(X_train, validation_data=Y_train, batch_size = batch_size,epochs=24,verbose=1, callbacks=[callback] )
