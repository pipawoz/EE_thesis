"""
*********************************************************************************
IMPORTS**************************************************************************
*********************************************************************************
"""

import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # elimino warnings de tensorflow-gpu


import logging
import random
from pathlib import Path

import hls4ml
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
import tensorflow as tf
from hls4ml.converters import convert_from_keras_model
from qkeras.qlayers import QActivation, QDense
from qkeras.quantizers import quantized_bits, quantized_relu
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.utils import to_categorical
from tensorflow_model_optimization.python.core.sparsity.keras import (
    prune, pruning_callbacks, pruning_schedule)
from tensorflow_model_optimization.sparsity.keras import strip_pruning

from qkeras.utils import model_save_quantized_weights

from callbacks import all_callbacks


"""
*********************************************************************************
PREPARO DATOS********************************************************************
*********************************************************************************
"""

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings
tf.get_logger().setLevel('ERROR')
os.environ['PATH'] = '/mnt/shared/Vivado/2019.2/bin:' + os.environ['PATH']


OUTPUT_FOLDER = "build"
Path("{}".format(OUTPUT_FOLDER)).mkdir(parents=True, exist_ok=True)

print("Carga Dataset")
file = np.load('./measurements_orig_64.npz')

DATA_LENGTH = file['data'].shape[1]

filedata = file['data'].reshape(len(file['data']), DATA_LENGTH)
labels = file['labels']
print(filedata.shape, labels.shape)

factor = 1 #Si uso _64 tengo que poner factor 1 # 512/2 = 256
data = filedata.reshape(len(file['data'])*factor, DATA_LENGTH//factor)

labels_64 = []
for label in labels:
    for i in range(factor):
        labels_64.append(label)    

labels_64 = np.array(labels_64)
print("Final Shape", data.shape, labels_64.shape)

category_labels = np.unique(labels_64)
print("Category Labels", category_labels)

# Paso labels a categorias por no poder usar strings
labels = pd.Categorical(labels_64, categories = category_labels).codes
print("Labels", labels)

test_size_len = int(len(data)*0.2)  # Separo 80/20

while True:
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = test_size_len, random_state = 100, 
                                                                        stratify = labels_64)

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape )

    np.save('{}/y_test.npy'.format(OUTPUT_FOLDER), test_labels)
    np.save('{}/X_test.npy'.format(OUTPUT_FOLDER), test_data)

    le = LabelEncoder()
    le.fit_transform(labels)
    np.save('{}/classes.npy'.format(OUTPUT_FOLDER), le.classes_, allow_pickle=True)

    """
    *********************************************************************************
    MODELO***************************************************************************
    *********************************************************************************
    """

    # Zybo 64-40-5
    def create_model(int_bits=2, n_bits=12):
        k_inic = random.choice(["lecun_uniform"]) #'glorot_uniform'
        model = Sequential()

        model.add(QDense(40, kernel_quantizer=quantized_bits(n_bits,int_bits,alpha=1), 
                        bias_quantizer=quantized_bits(n_bits,int_bits,alpha=1),
                        kernel_initializer=k_inic, 
                        kernel_regularizer=l1(0.0001),
                        name="input_dense"))
        model.add(QActivation(activation=quantized_relu(n_bits), name='Relu3'))

        model.add(QDense(5, kernel_quantizer=quantized_bits(n_bits,int_bits,alpha=1), 
                        bias_quantizer=quantized_bits(n_bits,int_bits,alpha=1),
                        kernel_initializer=k_inic, 
                        kernel_regularizer=l1(0.0001),
                        name="out_dense"))
        model.add(QActivation(activation=quantized_relu(n_bits), name='softmax'))
        opt = keras.optimizers.Adam()
        
        #loss = random.choice(["sparse_categorical_crossentropy", "mse", "categorical_crossentropy", "mean_absolute_error", "kld"])
        model.compile(loss="mse", optimizer=opt, metrics=["accuracy"])
        model.build(input_shape=(None, DATA_LENGTH//factor))
        return model


    """
    *********************************************************************************
    ENTRENAMIENTO********************************************************************
    *********************************************************************************
    """

    TOTAL_MODELS = 1  # Entreno 5 veces
    res = np.empty(TOTAL_MODELS)
    res[:] = np.nan
    max_res = 0
    BATCH_SIZE = 45
    EPOCHS = 200
    VALIDATION_SPLIT = 0.05  # Cantidad de datos que uso para validar entre epochs

    print("Comienzo Entrenamiento")

    mean_fit_time = 0
    mean_fit_acc = 0

    opt = keras.optimizers.Adam()
    pruning_params = {"pruning_schedule" : pruning_schedule.ConstantSparsity(0.75, begin_step=2000, frequency=100)}


    train = True
    if train:
        for i in range(TOTAL_MODELS):
            start_time = time.time()
    
            model = prune.prune_low_magnitude(create_model(int_bits=2, n_bits=10), **pruning_params)  #4181 2105
            model.compile(loss='mse', optimizer=opt, metrics=["accuracy"])
            model.summary()

            callbacks = all_callbacks(stop_patience = 1000,
                                    lr_factor = 0.5,
                                    lr_patience = 10,
                                    lr_epsilon = 0.000001,
                                    lr_cooldown = 2,
                                    lr_minimum = 0.0000001,
                                    outputDir = '{}'.format(OUTPUT_FOLDER))
            callbacks.callbacks.append(pruning_callbacks.UpdatePruningStep())

            history = model.fit(train_data, train_labels, epochs = EPOCHS, batch_size = BATCH_SIZE, validation_split=VALIDATION_SPLIT, verbose = 0,
                                callbacks = callbacks.callbacks)
            model_time = time.time() - start_time
            res[i] = model.evaluate(test_data, test_labels, batch_size = BATCH_SIZE, verbose = 0)[1]
            print('Iteracion: {}, Accuracy: {:.4f}, Time: {:2f}'.format(i+1, np.max(res[i]), model_time))

            mean_fit_time += model_time
            mean_fit_acc += np.max(res[i])
            
            model = strip_pruning(model)
            model.save('{}/KERAS_check_best_model.h5'.format(OUTPUT_FOLDER))

            if res[i] >= max_res:
                max_res = res[i]
                best_model = model
                best_history = history
                
        mean_fit_time /= TOTAL_MODELS
        mean_fit_acc /= TOTAL_MODELS
        model = best_model

    else:
        from qkeras.utils import _add_supported_quantized_objects
        from tensorflow.keras.models import load_model
        co = {}
        _add_supported_quantized_objects(co)
        model = load_model('{}/KERAS_check_best_model.h5'.format(OUTPUT_FOLDER), custom_objects=co)


    qweights = model_save_quantized_weights(model, "{}/model_qweights.h5".format(OUTPUT_FOLDER))

    y_keras = model.predict(test_data)
    np.save('{}/y_qkeras.npy'.format(OUTPUT_FOLDER), y_keras)
    print("Accuracy QKeras: {:.4f}\n".format(accuracy_score(np.argmax(test_labels, axis=1), np.argmax(y_keras, axis=1))))
    score_keras = accuracy_score(np.argmax(test_labels, axis=1), np.argmax(y_keras, axis=1))*100
    
    model.save('{}/model_{}_{:.2f}.h5'.format(OUTPUT_FOLDER, 1024//factor, score_keras))

    if score_keras <= 80:  # Repito el entrenamiento
        continue

    """
    *********************************************************************************
    CONVERSION A HLS*****************************************************************
    *********************************************************************************
    """

    input("Termino Entrenamiento. Enter para continuar.\n")
    
    print("Start HLS4ML")

    hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
    hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
    hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'

    config = hls4ml.utils.config_from_keras_model(model, granularity='name')

    precision = "ap_fixed<16,6>"
    resource = 4000

    config['Model'] = {}
    config['Model']['ReuseFactor'] = resource
    config['Model']['Strategy'] = 'Resource'
    config['Model']['Precision'] = precision

    for Layer in model.layers:
        if isinstance(Layer, tf.keras.layers.Flatten):
            config['LayerName'][Layer.name] = {}
        config['LayerName'][Layer.name]['Precision'] = precision
        config["LayerName"][Layer.name]["ReuseFactor"] = resource
        config["LayerName"][Layer.name]["Strategy"] = 'Resource'


    input_data = os.path.join(os.getcwd(), '{}/X_test.npy'.format(OUTPUT_FOLDER))
    output_predictions = os.path.join(os.getcwd(), '{}/y_qkeras.npy'.format(OUTPUT_FOLDER))

    input_data_tb = "{}/X_test.npy".format(OUTPUT_FOLDER)
    output_data_tb = "{}/y_test.npy".format(OUTPUT_FOLDER)

    io_type = 'io_parallel'  # io_parallel or io_stream

    hls_model = convert_from_keras_model(model=model, backend='VivadoAccelerator', io_type=io_type, board='zybo-z7010', part='xc7z010clg400-1',
                                        hls_config=config, output_dir="{}".format(OUTPUT_FOLDER), input_data_tb=input_data_tb, output_data_tb=output_data_tb)

    #hls_model = convert_from_keras_model(model=model, backend='VivadoAccelerator', io_type=io_type, board='zcu102',# part='xc7z010clg400-1',
    #                                    hls_config=config, output_dir="{}".format(OUTPUT_FOLDER), input_data_tb=input_data_tb, output_data_tb=output_data_tb)

    hls_model.compile()

    y_keras = model.predict(test_data)
    y_hls = hls_model.predict(test_data)

    np.save("{}/y_hls.npy".format(OUTPUT_FOLDER), y_hls)

    print("Accuracy QKeras: {:.4f}".format(accuracy_score(np.argmax(test_labels, axis=1), np.argmax(y_keras, axis=1))))
    print("Accuracy HLS4ML: {:.4f}".format(accuracy_score(np.argmax(test_labels, axis=1), np.argmax(y_hls, axis=1))))

    score_hls = accuracy_score(np.argmax(test_labels, axis=1), np.argmax(y_hls, axis=1))*100
    print("HLS_Score: {}%\n".format(score_hls))

    if score_hls >= 75:
        break
    

assert((hls_model.get_weights_data("input_dense", "kernel") == qweights["input_dense"]["weights"][0]).all())
assert((hls_model.get_weights_data("out_dense", "kernel") == qweights["out_dense"]["weights"][0]).all())

"""
*********************************************************************************
SINTESIS DE VIVADO***************************************************************
*********************************************************************************
"""

input("Termino HLS Compile. Enter para continuar\n")

hls_model.build(csim=False, synth=True, export=True)
hls4ml.report.read_vivado_report('{}/'.format(OUTPUT_FOLDER))

print("Accuracy QKeras: {:.4f}".format(accuracy_score(np.argmax(test_labels, axis=1), np.argmax(y_keras, axis=1))))
print("Accuracy HLS4ML: {:.4f}".format(accuracy_score(np.argmax(test_labels, axis=1), np.argmax(y_hls, axis=1))))

input("Termino HLS Build. Sigue Synth. Enter para continuar\n")

hls4ml.templates.VivadoAcceleratorBackend.make_bitfile(hls_model)


""" 
Zybo 64 con io_parallel y branch fifo
================================================================
== Utilization Estimates
================================================================
* Summary: 
+-----------------+---------+-------+-------+-------+-----+
|       Name      | BRAM_18K| DSP48E|   FF  |  LUT  | URAM|
+-----------------+---------+-------+-------+-------+-----+
|DSP              |        -|      -|      -|      -|    -|
|Expression       |        -|      -|     40|  19952|    -|
|FIFO             |        -|      -|      -|      -|    -|
|Instance         |      119|      2|  23236|  21676|    -|
|Memory           |        -|      -|      -|      -|    -|
|Multiplexer      |        -|      -|      -|    224|    -|
|Register         |        0|      -|   7995|    352|    -|
+-----------------+---------+-------+-------+-------+-----+
|Total            |      119|      2|  31271|  42204|    0|
+-----------------+---------+-------+-------+-------+-----+
|Available        |      120|     80|  35200|  17600|    0|
+-----------------+---------+-------+-------+-------+-----+
|Utilization (%)  |       99|      2|     88|    239|    0|
+-----------------+---------+-------+-------+-------+-----+



ZYBO 64 con io_parallel
================================================================
== Utilization Estimates
================================================================
* Summary: 
+-----------------+---------+-------+-------+-------+-----+
|       Name      | BRAM_18K| DSP48E|   FF  |  LUT  | URAM|
+-----------------+---------+-------+-------+-------+-----+
|DSP              |        -|      -|      -|      -|    -|
|Expression       |        -|      -|     40|  19970|    -|
|FIFO             |        -|      -|      -|      -|    -|
|Instance         |      119|      2|  23712|  21954|    -|
|Memory           |        -|      -|      -|      -|    -|
|Multiplexer      |        -|      -|      -|    224|    -|
|Register         |        0|      -|   8016|    352|    -|
+-----------------+---------+-------+-------+-------+-----+
|Total            |      119|      2|  31768|  42500|    0|
+-----------------+---------+-------+-------+-------+-----+
|Available        |      120|     80|  35200|  17600|    0|
+-----------------+---------+-------+-------+-------+-----+
|Utilization (%)  |       99|      2|     90|    241|    0|
+-----------------+---------+-------+-------+-------+-----+


ZYBO 256 con 4k y resource

================================================================
== Utilization Estimates
================================================================
* Summary: 
+-----------------+---------+-------+-------+-------+-----+
|       Name      | BRAM_18K| DSP48E|   FF  |  LUT  | URAM|
+-----------------+---------+-------+-------+-------+-----+
|DSP              |        -|      -|      -|      -|    -|
|Expression       |        -|      -|      0|     24|    -|
|FIFO             |      256|      -|  11319|  15400|    -|
|Instance         |       14|      6|  50842|  71272|    -|
|Memory           |        -|      -|      -|      -|    -|
|Multiplexer      |        -|      -|      -|     45|    -|
|Register         |        -|      -|      5|      -|    -|
+-----------------+---------+-------+-------+-------+-----+
|Total            |      270|      6|  62166|  86741|    0|
+-----------------+---------+-------+-------+-------+-----+
|Available        |      120|     80|  35200|  17600|    0|
+-----------------+---------+-------+-------+-------+-----+
|Utilization (%)  |      225|      7|    176|    492|    0|
+-----------------+---------+-------+-------+-------+-----+


Zybo con 128 y 4k
================================================================
== Utilization Estimates
================================================================
* Summary: 
+-----------------+---------+-------+-------+-------+-----+
|       Name      | BRAM_18K| DSP48E|   FF  |  LUT  | URAM|
+-----------------+---------+-------+-------+-------+-----+
|DSP              |        -|      -|      -|      -|    -|
|Expression       |        -|      -|      0|     24|    -|
|FIFO             |      128|      -|   5175|   6952|    -|
|Instance         |        7|      3|  22079|  39379|    -|
|Memory           |        -|      -|      -|      -|    -|
|Multiplexer      |        -|      -|      -|     45|    -|
|Register         |        -|      -|      5|      -|    -|
+-----------------+---------+-------+-------+-------+-----+
|Total            |      135|      3|  27259|  46400|    0|
+-----------------+---------+-------+-------+-------+-----+
|Available        |      120|     80|  35200|  17600|    0|
+-----------------+---------+-------+-------+-------+-----+
|Utilization (%)  |      112|      3|     77|    263|    0|
+-----------------+---------+-------+-------+-------+-----+

PYNQ 512
================================================================
* Summary: 
+-----------------+---------+-------+--------+-------+-----+
|       Name      | BRAM_18K| DSP48E|   FF   |  LUT  | URAM|
+-----------------+---------+-------+--------+-------+-----+
|DSP              |        -|      -|       -|      -|    -|
|Expression       |        -|      -|       0|     24|    -|
|FIFO             |      512|      -|   23607|  35644|    -|
|Instance         |       26|      6|   88629|  82791|    -|
|Memory           |        -|      -|       -|      -|    -|
|Multiplexer      |        -|      -|       -|     45|    -|
|Register         |        -|      -|       5|      -|    -|
+-----------------+---------+-------+--------+-------+-----+
|Total            |      538|      6|  112241| 118504|    0|
+-----------------+---------+-------+--------+-------+-----+
|Available        |      280|    220|  106400|  53200|    0|
+-----------------+---------+-------+--------+-------+-----+
|Utilization (%)  |      192|      2|     105|    222|    0|
+-----------------+---------+-------+--------+-------+-----+
"""
