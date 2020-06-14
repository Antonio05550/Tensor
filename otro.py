import tensorflow as tf
from scipy.stats import stats
from tensorflow import keras, optimizers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.layers import Conv2D

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#---------------------------------- Conjutunto de datos y Preprocesamiento ----------------------------------

def leer_datos(ruta):
    columnas = ['IdUsuario', 'actividad', 'momentoCap', 'a_x', 'a_y', 'a_z']
    datos = pd.read_csv(ruta,header = None, names = columnas)
    return datos

def normalizacion_caract(datos):
    #return (datos - np.mean(datos)) / np.std(datos)
    media = np.mean(datos,axis = 0)
    desviacion_estandar = np.std(datos,axis = 0)
    return (datos - media)/desviacion_estandar

def ventana(datos, size):
    start = 0
    while start < datos.count():
        yield int(start), int(start + size)
        start += (size / 2)

def plot_senal(ax, momento, datos, eje):
    ax.plot(momento, datos)
    ax.set_title(eje)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(datos) - np.std(datos), max(datos) + np.std(datos)])
    ax.set_xlim([min(momento), max(momento)])
    ax.grid(True)

def plot_actividad(actividad,datos):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows = 3, figsize = (15, 10), sharex = True)
    plot_senal(ax0, datos['momentoCap'], datos['a_x'], 'X Acelerometro')
    plot_senal(ax1, datos['momentoCap'], datos['a_y'], 'Y Acelerometro')
    plot_senal(ax2, datos['momentoCap'], datos['a_z'], 'Z Acelerometro')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(actividad)
    plt.subplots_adjust(top=0.90)
    plt.show()

#-------------------------------------- Leemos el Conjutunto de datos --------------------------------------

conjuntoDatos = leer_datos('DB.txt')
conjuntoDatos.dropna(axis=0, how='any', inplace=True)

#--------------- Normalizamos las caracteristicas del conjunto de datos para el entrenamiento ---------------

conjuntoDatos['a_x'] = normalizacion_caract(conjuntoDatos['a_x'])
conjuntoDatos['a_y'] = normalizacion_caract(conjuntoDatos['a_y'])
conjuntoDatos['a_z'] = normalizacion_caract(conjuntoDatos['a_z'])

#---------------------------------- Sub-Conjutunto de datos para visualizar ---------------------------------

for actividad in np.unique(conjuntoDatos["actividad"]):
    subConjunto = conjuntoDatos[conjuntoDatos["actividad"] == actividad][:180]
    plot_actividad(actividad,subConjunto)

#--------------------------- Preparamos el Conjutunto de datos para el modelo CNN ---------------------------

def ventana(datos, size):
    start = 0
    while start < datos.count():
        yield int(start), int(start + size)
        start += (size / 2)

def segment_signal(datos,ventana_size = 90):
    segmentos = np.empty((0,ventana_size,3))
    etiquetas = np.empty((0))
    for (start, end) in ventana(datos["momentoCap"], ventana_size):
        x = datos["a_x"][start:end]
        y = datos["a_y"][start:end]
        z = datos["a_z"][start:end]
        if(len(conjuntoDatos["momentoCap"][start:end]) == ventana_size):
            segmentos = np.vstack([segmentos,np.dstack([x,y,z])])
            etiquetas = np.append(etiquetas,stats.mode(datos["actividad"][start:end])[0][0])
    return segmentos, etiquetas
segmentos, etiquetas = segment_signal(conjuntoDatos)
etiquetas = np.asarray(pd.get_dummies(etiquetas), dtype = np.int8)

#------------------------------------------ Creamos el Modelo CNN  ------------------------------------------

numFilas = segmentos.shape[1]
numColumnas = segmentos.shape[2]
numCanales = 1
numFiltros = 128  # Numero de filtros de la capa Conv2D
# Tamaño del kernel de la capa Conv2D
kernelSize1 = 2
# Tamaño maximo de la ventana de agrupacion
ventanaAgru = 2
# Numero de filtros en capas conectadas
numNeuronasFCL1 = 128
numNeuronasFCL2 = 128
# Porcentaje de división de datos para pruebas y validacion
porcentajePrueba = 0.8
# Numero de Epocas
Epocas = 10
# Tabaño prueba
numPruebas = 32
# Numero ttotal de clases
numClases = etiquetas.shape[1]
# Porcentaje de abandono para la capa
poncentajeAbandono = 0.5
# Datos nuevos para la red
segmentosNuevos = segmentos.reshape(segmentos.shape[0], numFilas, numColumnas)
#segmentosNuevos = segmentos.reshape(len(segmentos), 1,90, 3)
# División de datos para entrenamiento y pruebas
train_test_split = np.random.rand(len(segmentosNuevos)) < 0.70
train_x = segmentosNuevos[train_test_split]
train_y = etiquetas[train_test_split]
test_x = segmentosNuevos[~train_test_split]
test_y = etiquetas[~train_test_split]
print('x_train shape:', train_x.shape)
print('y_train shape:', train_y.shape)
model_m = tf.keras.models.Sequential()
model_m.add(tf.keras.layers.InputLayer(input_shape=(numFilas,numColumnas)))
model_m.add(tf.keras.layers.Reshape(target_shape=(numFilas,numColumnas,1)))
model_m.add(tf.keras.layers.Dense(100, activation='relu'))
model_m.add(tf.keras.layers.Dense(100, activation='relu'))

model_m.add(tf.keras.layers.Dense(100, activation='relu'))
model_m.add(tf.keras.layers.Flatten())
model_m.add(tf.keras.layers.Dense(numClases, activation='softmax'))
print(model_m.summary())

adam = optimizers.Adam(lr=0.001, decay=1e-6)
model_m.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model_m.fit(train_x, train_y, validation_split=1 - porcentajePrueba, epochs=10, batch_size=numPruebas)
score = model_m.evaluate(test_x, test_y, verbose=0)

converter = tf.lite.TFLiteConverter.from_keras_model(model_m)
tflite_model = converter.convert()

# Save the TF Lite model.
with tf.io.gfile.GFile('model.tflite', 'wb') as f:
  f.write(tflite_model)
