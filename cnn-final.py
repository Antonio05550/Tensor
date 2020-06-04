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
#segmentosNuevos = segmentos.reshape(len(segmentos), 1,90, 3)

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
segmentosNuevos = segmentos.reshape(segmentos.shape[0], numFilas, numColumnas, 1)
# División de datos para entrenamiento y pruebas
trainSplit = np.random.rand(len(segmentosNuevos)) < porcentajePrueba
trainX = segmentosNuevos[trainSplit]
testX = segmentosNuevos[~trainSplit]
trainX = np.nan_to_num(trainX)
testX = np.nan_to_num(testX)
trainY = etiquetas[trainSplit]
testY = etiquetas[~trainSplit]

# Definimos el modelo
model = Sequential()
# Agregamos la primera capa convolucionial con 32 filtros y tamaño kernel de 5 por 5, utilizando el rectificador
# como funcion de activacion
model.add(
    Conv2D(numFiltros, (kernelSize1, kernelSize1), input_shape=(numFilas, numColumnas, 1), activation='relu'))
# Agregamos una capa de MaxPooling
model.add(MaxPooling2D(pool_size=(ventanaAgru, ventanaAgru), padding='valid'))
# Agregamos una capa de abandono para la regularizacion y evitar el ajuste excesivo
model.add(Dropout(poncentajeAbandono))
# Aplanamos la salida
model.add(Flatten())
# Agregamos la primera capa totalmente conectada con 256 salidas
model.add(Dense(numNeuronasFCL1, activation='relu'))
# Agregamos la segunda capa de 128 salidas toralmenre conectadas
model.add(Dense(numNeuronasFCL2, activation='relu'))
# Agregamos la capa softmax para la clasificacion
model.add(Dense(numClases, activation='softmax'))

#------------------------------------------ Ajustamos el Modelo CNN  ------------------------------------------

# Compilamos el modelo para generar un modelo
adam = optimizers.Adam(lr=0.001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

for layer in model.layers:
    print(layer.name)

model.fit(trainX, trainY, validation_split=1 - porcentajePrueba, epochs=10, batch_size=numPruebas)
score = model.evaluate(testX, testY, verbose=0)

converter = tf.lite.TFLiteConverter.from_saved_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()
open("converted_model.tflite", "wb").write(quantized_model)

# Convert Keras model to TF Lite format.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_float_model = converter.convert()

# Show model size in KBs.
float_model_size = len(tflite_float_model) / 1024
print('Float model size = %dKBs.' % float_model_size)

# Re-convert the model to TF Lite using quantization.
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

# Show model size in KBs.
quantized_model_size = len(tflite_quantized_model) / 1024
print('Quantized model size = %dKBs,' % quantized_model_size)
print('which is about %d%% of the float model size.'\
      % (quantized_model_size * 100 / float_model_size))

# Save the quantized model to file to the Downloads directory
f = open('recoact.tflite', "wb")
f.write(tflite_quantized_model)
f.close()
