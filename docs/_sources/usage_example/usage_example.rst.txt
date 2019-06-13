##############
Usage Examples
##############

.. code:: ipython3

    import tensorflow.keras as keras
    import tensorflow.keras.layers as layers
    import tensorflow.keras.backend as K
    
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    
    import os, sys
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    lib_path = os.path.abspath("..")
    sys.path.append(lib_path)
    
    import keras_one_cycle_clr as ktool

Generating dataset
==================

.. code:: ipython3

    x = np.linspace(-1, 1, 1001)
    
    np.random.seed(99)
    
    y = x**3 - 0.5 *x**2 - x + np.random.randn(*x.shape)*0.1 + 1*np.exp(-np.abs(x)) \
        - np.round((x+1)/2)*np.sqrt(3.0-x**2) + np.round((x+1-0.5)/2)*np.power(3.0-x, 0.33)
    
    # prepare a test set
    validation_fraction = 0.2
    valid_ind = int(np.ceil( (1-validation_fraction)*x.size))
    
    ind = np.arange(0, x.size)
    np.random.shuffle(ind)
    
    x_train = np.take(x, ind[0: valid_ind])
    y_train = np.take(y, ind[0: valid_ind])
    x_test = np.take(x, ind[valid_ind:])
    y_test = np.take(y, ind[valid_ind:])
    
    plt.figure(figsize=(8, 6))
    
    plt.plot(x_train, y_train, '.')
    plt.plot(x_test, y_test, '.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()



.. image:: output_2_0.png


Define a model
==============

.. code:: ipython3

    def build_model(l2_reg=0.0):
        input = layers.Input(shape=(1,))
        x = layers.Dense(20, activation="relu", kernel_initializer="he_normal",
                        kernel_regularizer=keras.regularizers.l2(l2_reg))(input)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(1)(x)
        model = keras.Model(inputs=input, outputs=x)
        return model

.. code:: ipython3

    K.clear_session()
    model = build_model()


.. parsed-literal::

    WARNING:tensorflow:From /Users/vikube/miniconda3/envs/ml/lib/python3.6/site-packages/tensorflow/python/ops/resource_variable_ops.py:435: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.


.. code:: ipython3

    def reset_model(model):
        optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.95)
        model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=[keras.metrics.mean_squared_error])
        model.load_weights("demo_stage_0.hdf5")

.. code:: ipython3

    reset_model(model)


.. parsed-literal::

    WARNING:tensorflow:From /Users/vikube/miniconda3/envs/ml/lib/python3.6/site-packages/tensorflow/python/keras/utils/losses_utils.py:170: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.


.. code:: ipython3

    # model.save_weights("demo_stage_0.hdf5")
    # model.save("demo_stage_0.h5")

.. code:: ipython3

    batch_size = 16

LrRT
====

.. code:: ipython3

    K.clear_session()
    model = keras.models.load_model("demo_stage_0.h5")

.. code:: ipython3

    lrrf_cb = ktool.LrRangeTest(lr_range = (1e-3, 10),
                     wd_list = [],
                     steps=100,
                     batches_per_step=5,
                     threshold_multiplier=5.0,
                     validation_data=None,
                     batches_per_val = 5,
                     verbose=True)
    
    n_epoch = lrrf_cb.find_n_epoch(x_train, batch_size)
    
    model.fit(x=x_train, y=y_train, verbose=0,
              epochs=n_epoch,
              batch_size=batch_size,
              validation_data=None,
              callbacks=[lrrf_cb])



.. code:: ipython3

    lrrf_cb.plot()



.. image:: output_13_0.png


.. code:: ipython3

    lrrf_wd_cb = ktool.LrRangeTest(lr_range = (1e-3, 10),
                     wd_list = [0, 1e-4, 1e-2],
                     steps=100,
                     batches_per_step=5,
                     threshold_multiplier=4,
                     validation_data=(x_test, y_test),
                     batches_per_val = 10,
                     verbose=True)

.. code:: ipython3

    n_epoch = lrrf_wd_cb.find_n_epoch(x_train, batch_size)
    
    model.fit(x=x_train, y=y_train, verbose=0,
              epochs=n_epoch,
              batch_size=batch_size,
              validation_data=None,
              callbacks=[lrrf_wd_cb])



.. code:: ipython3

    lrrf_wd_cb.plot(set='valid')



.. image:: output_16_0.png


One Cycle - 20-epoch
====================

.. code:: ipython3

    ktool.utils.reset_keras()
    model = keras.models.load_model("demo_stage_0.h5")
    reset_model(model)

.. code:: ipython3

    K.set_value(model.optimizer.decay, 1e-4)

.. code:: ipython3

    ocp = ktool.OneCycle(lr_range=(1e-2/5, 1e-2),
                        momentum_range=(0.95, 0.85),
                        verbose=False)
    
    ocp_hist = model.fit(x_train, y_train,
                         epochs=40,
                         validation_data=(x_test, y_test),
                         verbose=2,
                         callbacks=[ocp])


.. code:: ipython3

    ktool.utils.plot_from_history(ocp_hist)



.. image:: output_21_0.png


Small constant learning rate: 0.005
===================================

.. code:: ipython3

    K.clear_session()
    model = keras.models.load_model("demo_stage_0.h5")
    reset_model(model)

.. code:: ipython3

    K.set_value(model.optimizer.momentum, 0.95)
    K.set_value(model.optimizer.lr, 0.005)

.. code:: ipython3

    const_lr_hist = model.fit(x_train, y_train,
                             epochs=40,
                             validation_data=(x_test, y_test),
                             verbose=2)


.. code:: ipython3

    const_lr_hist = model.history
    ktool.utils.plot_from_history(const_lr_hist)



.. image:: output_26_0.png


CLR
===

.. code:: ipython3

    K.clear_session()
    model = keras.models.load_model("demo_stage_0.h5")
    reset_model(model)

.. code:: ipython3

    clr_cb = ktool.CLR(cyc=3,
                       lr_range=(1e-2/5, 1e-2),
                       momentum_range=(0.95, 0.85),
                       verbose=False,
                       amplitude_fn=lambda x: np.power(1.0/3, x))

.. code:: ipython3

    clr_hist = model.fit(x_train, y_train,
                         epochs=60,
                         validation_data=(x_test, y_test),
                         verbose=2,
                         callbacks=[clr_cb])



.. code:: ipython3

    ktool.utils.plot_from_history(clr_hist)



.. image:: output_31_0.png


Comparing validation test
=========================

.. code:: ipython3

    hists = [ocp_hist, clr_hist, const_lr_hist]
    legends = ['ocp', 'clr', 'const']
    
    plt.figure(figsize=(8, 6))
    
    for i, hist in enumerate(hists):
        plt.plot(hist.epoch, hist.history['val_loss'], label=legends[i])
        
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('val_loss')
    
    plt.show()



.. image:: output_33_0.png


