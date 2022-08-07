import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar100.load_data()
x_train=(x_train)/255.0
x_test=(x_test)/255.0
x_train.shape,y_train.shape,x_test.shape,y_test.shape

x_train[0].min(),x_train[0].max()

class_labels={
0: 'apple',
1: 'aquarium_fish',
2: 'baby',
3: 'bear',
4: 'beaver',
5: 'bed',
6: 'bee',
7: 'beetle',
8: 'bicycle',
9: 'bottle',
10: 'bowl',
11: 'boy',
12: 'bridge',
13: 'bus',
14: 'butterfly',
15: 'camel',
16: 'can',
17: 'castle',
18: 'caterpillar',
19: 'cattle',
20: 'chair',
21: 'chimpanzee',
22: 'clock',
23: 'cloud',
24: 'cockroach',
25: 'couch',
26: 'crab',
27: 'crocodile',
28: 'cup',
29: 'dinosaur',
30: 'dolphin',
31: 'elephant',
32: 'flatfish',
33: 'forest',
34: 'fox',
35: 'girl',
36: 'hamster',
37: 'house',
38: 'kangaroo',
39: 'keyboard',
40: 'lamp',
41: 'lawn_mower',
42: 'leopard',
43: 'lion',
44: 'lizard',
45: 'lobster',
46: 'man',
47: 'maple_tree',
48: 'motorcycle',
49: 'mountain',
50: 'mouse',
51: 'mushroom',
52: 'oak_tree',
53: 'orange',
54: 'orchid',
55: 'otter',
56: 'palm_tree',
57: 'pear',
58: 'pickup_truck',
59: 'pine_tree',
60: 'plain',
61: 'plate',
62: 'poppy',
63: 'porcupine',
64: 'possum',
65: 'rabbit',
66: 'raccoon',
67: 'ray',
68: 'road',
69: 'rocket',
70: 'rose',
71: 'sea',
72: 'seal',
73: 'shark',
74: 'shrew',
75: 'skunk',
76: 'skyscraper',
77: 'snail',
78: 'snake',
79: 'spider',
80: 'squirrel',
81: 'streetcar',
82: 'sunflower',
83: 'sweet_pepper',
84: 'table',
85: 'tank',
86: 'telephone',
87: 'television',
88: 'tiger',
89: 'tractor',
90: 'train',
91: 'trout',
92: 'tulip',
93: 'turtle',
94: 'wardrobe',
95: 'whale',
96: 'willow_tree',
97: 'wolf',
98: 'woman',
99: 'worm'
}

def show_images(images,labels,preds=False):
    fig=plt.figure()
    fig.set_figwidth(20)
    fig.set_figheight(10)
    plt.style.use('ggplot')
    for idx in range(images.shape[0]):
        plt.subplot(4,8,idx+1)
        img=images[idx]
        plt.imshow(img)
        if preds:
            plt.title(class_labels[np.argmax(labels[idx])])
        else:
            plt.title(class_labels[labels[idx].item()])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

from tensorflow.keras.models import Sequential

data_augmentation=Sequential([
    tf.keras.layers.RandomFlip(mode='horizontal'),
    tf.keras.layers.RandomTranslation(width_factor=.4,height_factor=.4),
    tf.keras.layers.RandomContrast(.15),
    tf.keras.layers.RandomRotation(.06),
    tf.keras.layers.RandomZoom(.2),
])

train_data=tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_data=train_data.cache()
train_data=train_data.shuffle(50000)
train_data=train_data.batch(128)
train_data=train_data.map(lambda x,y:(data_augmentation(x),y))
train_data=train_data.prefetch(64)

test_data=tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_data=test_data.batch(32)
test_data=test_data.prefetch(16)

train_iterator=train_data.as_numpy_iterator()

images,labels=train_iterator.next()
show_images(images[:32],labels[:32])

from tensorflow.keras.layers import Conv2D,Dense,Flatten,Dropout,BatchNormalization,MaxPool2D
from tensorflow.keras import Input,Model

def build_model():
    inputs=Input(shape=(32,32,3),name='input_layer')
    x=Conv2D(64,(3,3),1,padding='same',activation='relu')(inputs)
    x=BatchNormalization()(x)
    x=Conv2D(64,(3,3),1,padding='same',activation='relu')(x)
    x=BatchNormalization()(x)

    x=Conv2D(128,(3,3),1,padding='same',activation='relu')(x)
    x=BatchNormalization()(x)
    x=Conv2D(128,(3,3),1,padding='same',activation='relu')(x)
    x=BatchNormalization()(x)
    x=MaxPool2D()(x)

    x=Conv2D(192,(3,3),1,padding='same',activation='relu')(x)
    x=BatchNormalization()(x)
    x=Conv2D(192,(3,3),1,padding='same',activation='relu')(x)
    x=BatchNormalization()(x)
    x=MaxPool2D()(x)

    x=Flatten()(x)
    x=Dropout(.2)(x)
    x=Dense(512,activation='relu')(x)
    x=BatchNormalization()(x)
    x=Dropout(.2)(x)
    outputs=Dense(100,activation='softmax')(x)

    model=Model(inputs=inputs,outputs=outputs,name='cifar100_model')
    return model

model=build_model()
model.summary()

model=tf.keras.models.load_model(os.path.join('models','cifar10_model.h5'))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# history=model.fit(train_data,epochs=30,validation_data=test_data,callbacks=[tf.keras.callbacks.TensorBoard(log_dir='logs')])

df=pd.DataFrame(history.history)
df.to_csv('model_loss.csv',index=False)

plt.style.use('fivethirtyeight')
df=pd.read_csv('model_loss.csv')
plt.figure(figsize=(20,5))
fig,(ax1,ax2)=plt.subplots(nrows=1,ncols=2,figsize=(20,7))
ax1.plot(df['loss'],label='Loss')
ax1.plot(df['val_loss'],label='Val_loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Value')
ax1.set_title('Loss Trend')
ax1.legend()

ax2.plot(df['accuracy'],label='Accuracy')
ax2.plot(df['val_accuracy'],label='Val_accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Value')
ax2.set_title('Accuracy Trend')
ax2.legend()

plt.show()

# tf.keras.models.save_model(model,os.path.join('models','cifar100_model.h5'))
# model.save('./metadata/')

"""## Predictions on training dataset"""

images,labels=train_iterator.next()
pred=model.predict(images)
show_images(images[:32],pred[:32],True)

"""## Predictions on testing dataset"""

loss,acc=model.evaluate(x_test,y_test,verbose=0)
print(f'Model Loss: {loss:.4f}')
print(f'Model Accuracy: {acc*100:.3f} %')

preds=model.predict(x_test)

indexes=np.random.randint(0,9999,size=(32,))
images=np.expand_dims(x_test[indexes[0]],axis=0)
pred=np.expand_dims(preds[indexes[0]],axis=0)
for i in range(1,len(indexes)):
    images=np.concatenate([images,np.expand_dims(x_test[indexes[i]],axis=0)],axis=0)
    pred=np.concatenate([pred,np.expand_dims(preds[indexes[i]],axis=0)],axis=0)
show_images(images,pred,True)

