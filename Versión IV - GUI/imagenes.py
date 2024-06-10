# Procedemos a importar las librerias necesarias para desarrollar el proyecto.

import numpy as np
import os
from tensorflow.keras import Sequential
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten, Reshape, Conv2DTranspose, UpSampling2D, Input,MaxPooling2D
from tensorflow.keras import Model
import seaborn as sn
from tensorflow.keras.optimizers import SGD
import warnings
warnings.filterwarnings("ignore")

from keras.src.legacy.preprocessing.image import ImageDataGenerator
#from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------

# Definimos las rutas donde se alojan los datos.
train      = 'seg_train/seg_train/'
validation = 'seg_test/seg_test/'
test       = 'seg_pred/'
test1  ='img_subida/'
test2  ='subidausuario/'

#-------------------------------------------------------------------------------
# Utilizamos la libreria proporcionada por Keras para leer las imagenes del DataSet.
train_datagen = ImageDataGenerator(rescale = 1./255)

validation_datagen = ImageDataGenerator(rescale = 1./255)
validation_datagen1 = ImageDataGenerator(rescale = 1./255)


train_generator = train_datagen.flow_from_directory(train,
                                                   batch_size=32,
                                                   target_size = (150, 150),
                                                   class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(validation,
                                                   batch_size=32,
                                                   shuffle=False,
                                                   target_size = (150, 150),
                                                   class_mode='categorical')

"""validation_generator1 = validation_datagen1.flow_from_directory(test1,
                                                   batch_size=32,
                                                   shuffle=False,
                                                   target_size = (150, 150),
                                                   class_mode='categorical')"""


test_datagen = ImageDataGenerator(rescale = 1./255)
test_generator = test_datagen.flow_from_directory(test, target_size = (150, 150), batch_size = 1)

test_datagen1 = ImageDataGenerator(rescale = 1./255)
test_generator1 = test_datagen1.flow_from_directory(test1,target_size = (150, 150), batch_size = 1)

#---------------------------------------------------------------------------------------------------
# Obtenemos los nombres de las clases.
class_names = train_generator.class_indices
print(class_names)
label=list(class_names.keys())

#-----------------------------------------------------------------------------------------------------
"""
# Imprimimos algunas imagenes que se utilizarán en el entrenamiento con su respectivo rótuolo.
for img_batch,label_batch in train_generator:
    print(img_batch.shape)
    plt.figure(figsize=(20,10))
    for ix in range(32):
        sub = plt.subplot(4, 8, ix + 1)
        plt.imshow(img_batch[ix])
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(label[np.argmax(label_batch[ix])])
    break
plt.show()
"""
#-------------------------------------------------------------------------------------------------------
# Definimos el método que creará la arquitectura de la red neuronal.
def build_model(input_shape = (150, 150, 3)):
    Model = Sequential()
    Model.add(Conv2D(64, kernel_size = (3, 3), strides = (2, 2), padding = 'same', activation = 'relu', input_shape = input_shape))
    Model.add(Conv2D(128, kernel_size = (3, 3), strides = (2, 2), padding = 'same', activation = 'relu'))
    Model.add(MaxPool2D())
    Model.add(Conv2D(256, kernel_size = (3, 3), strides = (2, 2), padding = 'same', activation = 'relu'))
    Model.add(Conv2D(512, kernel_size = (3, 3), strides = (2, 2), padding = 'same', activation = 'relu'))
    Model.add(MaxPool2D())
    Model.add(Flatten())
    Model.add(Dropout(0.2))
    Model.add(Dense(1048, activation = 'relu'))
    Model.add(Dropout(0.2))
    Model.add(Dense(len(class_names), activation = 'softmax'))

    Model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return Model
#--------------------------------------------------------------------------------------------------------


# Creamos la red neuronal para que se entrene utilizando el TPU.
model = build_model()
model.summary()

#----------------------------------------------------------------------------------------------------------
# Cargamos los parametros de las neuronas.
model.load_weights('model_weights_cnn.h5')

#----------------------------------------------------------------------------------------------------------
error = model.evaluate(validation_generator)
#----------------------------------------------------------------------------------------------------------


## aqui va lo de la matriz de confusion


# Preparamos los datos para graficarlos en una matriz de confusión.
"""predictions = model.predict_generator(validation_generator)
predicted_classes = np.argmax(predictions, axis=1)"""

predictions=model.predict(validation_generator) 
predicted_classes=np.argmax(predictions,axis=1)


# Graficamos la matriz de confusión.
"""
CM = confusion_matrix(validation_generator.classes,predicted_classes)
ax = plt.axes()
sn.heatmap(CM, annot=True,
           annot_kws={"size": 10},
           fmt="d",#mostrar números de manera no exponencial.
           xticklabels=class_names,
           yticklabels=class_names,
           linewidths=.5,
           ax = ax)
ax.set_title('Confusion matrix')
plt.show()
"""
#--------------------------------------------------------------------
"""
# Imprimimos algunas imagenes que se utilizarán en el entrenamiento con su respectivo rótuolo.
for img_batch,label_batch in train_generator:
    print(img_batch.shape)
    plt.figure(figsize=(20,10))
    for ix in range(32):
        sub = plt.subplot(4, 8, ix + 1)
        plt.imshow(img_batch[ix])
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(label[np.argmax(label_batch[ix])])
    break
plt.show()
"""
#------------------------------------------------------

"""
# Ponemos a prueba el modelo con los datos para testear.
index = 1
plt.figure(figsize=(20,10))
row = 4
col = 8
for img_batch,_ in test_generator:
    sub = plt.subplot(row, col, index)
    plt.imshow(img_batch[0])
    plt.xticks([])
    plt.yticks([])
    pred = model.predict(img_batch)
    class_key = np.argmax(pred)
    prob = np.max(pred) * 100
    plt.ylabel('{:.2f}%'.format(prob))
    plt.xlabel(label[np.argmax(pred)])
    index = index + 1
    if index > row * col:
    #if index > 1:
        break
plt.show()
"""
#-----------------------------------------------------------

# de la prediccion le digo que solo muestre los de mares
# esto puede servir en el caso de que quiera mostrarle al usuario una cantidad determinada de imagenes de cierto tipo
# ej: el usuario ingresa una imagen y el sistema le devuelve imagenes similares
#EJEMPLO CON VOZ: SI SE INGRESA LA PALABRA MAR, DEBERIA MOSTRAR SOLO DE MARES, SI HAY CANTIDAD, SOLO ESA CANTIDAD
# AQUI SE MUESTRAN SOLO 4 IMAGENES DE MARES,
#importante: se deberia descartar una entrada no valida porque si no se hace infinito....ej: FOREST en lugar de forest
index = 1
plt.figure(figsize=(20,10))
row = 2
col = 2
#for img_batch,_ in test_generator:
for img_batch,_ in test_generator1:
     sub = plt.subplot(row, col, index)
     plt.imshow(img_batch[0])
     plt.xticks([])
     plt.yticks([])
     pred = model.predict(img_batch)
     class_key = np.argmax(pred)
     prob = np.max(pred) * 100
#if(label[np.argmax(pred)]=="forest"):
     sub = plt.subplot(row, col, index)
     plt.imshow(img_batch[0])
     plt.xticks([])
     plt.yticks([])
     plt.ylabel('{:.2f}%'.format(prob))
     plt.xlabel(label[np.argmax(pred)])
     index = index + 1
     if index > row * col:
    #if index > 1:
      break
#plt.show()
#----------------------------------------------------------------------------------------------------------
#!pip install gTTS
from gtts import gTTS

#-----------------------------------------------------------------------------------------------------
test_datagen2 = ImageDataGenerator(rescale = 1./255)
test_generator2 = test_datagen2.flow_from_directory(test1,target_size = (150, 150), batch_size = 1)

plt.figure(figsize=(2,5))
plt.imshow(img_batch[0])
plt.xticks([])
plt.yticks([])
pred = model.predict(img_batch)
class_key = np.argmax(pred)
prob = np.max(pred) * 100
class_name = label[class_key]
plt.ylabel('{:.2f}%'.format(prob))
plt.xlabel(label[np.argmax(pred)])
# Clasificación de la imagen
#Generación del texto a hablar
if(prob<75): duda="no estoy seguro"
else: duda=""
text_to_speak = f'La imagen ha sido clasificada como {duda}  {class_name} con una probabilidad del {prob:.2f}%.'

plt.show()

# Clasificación de la imagen
pred = model.predict(img_batch)
class_key = np.argmax(pred)
prob = np.max(pred) * 100
class_name = label[class_key]

# Generación del texto a hablar
text_to_speak = f'La imagen pertenece a la Región Turistica de {class_name} con una probabilidad del {prob:.2f}%.'
text_to_speak_capitales= f'las ciudades de las provincias son estas...'


#------------------------------------------------------------------------
# Generación del archivo de audio
tts = gTTS(text=text_to_speak, lang='es')  # 'es' para español
tts.save('result_audio.mp3')

#------------------------------------------------------------------------

from IPython.display import Audio

# Reproducción del audio
Audio("result_audio.mp3", autoplay=True)








































