import tensorflow as tf
from tensorflow import keras
from dataPreprocessing import *
import os
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import compute_class_weight

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import f1_score, recall_score, precision_score
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def makeModel(train_features, init_bias=None):
    metrics = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]
    output_bias = init_bias
    print('output bias ', output_bias)
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
        print('output bias ', output_bias)
    model = keras.Sequential([
        keras.layers.Dense(
            6, activation='relu', use_bias=True,
            input_shape=(train_features.shape[-1],)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid',
                           bias_initializer=output_bias),
    ])

    model.compile(
        optimizer= keras.optimizers.SGD(learning_rate=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[get_f1])

    return model

def runKeras(oversample_data = 0):
    mpl.rcParams['figure.figsize'] = (12, 10)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    df = getTrainingData('Train.csv', False, False, True)
    neg, pos = np.bincount(df['Exited'])
    total = neg + pos
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))

    train_df, val_df = train_test_split(df, test_size=0.2)
    test_df = getTestingData(False, True)
    test_labels = np.array(test_df.pop('Exited'))

    train_labels = np.array(train_df.pop('Exited'))
    bool_train_labels = train_labels != 0
    val_labels = np.array(val_df.pop('Exited'))


    train_features = np.array(train_df)
    val_features = np.array(val_df)
    test_features = np.array(test_df)


    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)

    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    print('Training labels shape:', train_labels.shape)
    print('Validation labels shape:', val_labels.shape)
    print('Test labels shape:', test_labels.shape)
    print('Training features shape:', train_features.shape)
    print('Validation features shape:', val_features.shape)
    print('Test features shape:', test_features.shape)

    EPOCHS = 60
    BATCH_SIZE = 3
    init_bias = np.log([pos/neg])
    print(init_bias, ' initial bias')

    # metrics = Metrics(file_path)
    # callbacks_list = [metrics]

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        verbose=1,
        patience=20,
        mode='min',
        restore_best_weights=True)

    model = makeModel(train_features, init_bias)
    model.summary()
    model.predict(train_features)
    results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
    print("Loss: {:0.4f}".format(results[0]))
    initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
    model.save_weights(initial_weights)


    # weight_for_0 = (1 / neg) * (total) / 2.0
    # weight_for_1 = (1 / pos) * (total) / 2.0
    #
    # class_weight = {0: weight_for_0, 1: weight_for_1}
    # print('Weight for class 0: {:.2f}'.format(weight_for_0))
    # print('Weight for class 1: {:.2f}'.format(weight_for_1))

    class_weights = compute_class_weight('balanced', np.unique(train_labels), train_labels)
    class_weights = {i: class_weights[i] for i in range(2)}
    print(class_weights)


    if oversample_data == 0:

        weighted_model = makeModel(train_features)
        weighted_model.load_weights(initial_weights)

        weighted_history = weighted_model.fit(
            train_features,
            train_labels,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            # callbacks=[early_stopping],
            validation_data=(val_features, val_labels),
            # The class weights go here
            class_weight=class_weights)

        test_predictions_weighted = weighted_model.predict(test_features, batch_size=BATCH_SIZE)
        weighted_results = weighted_model.evaluate(test_features, test_labels,
                                                   batch_size=BATCH_SIZE, verbose=0)
        for name, value in zip(weighted_model.metrics_names, weighted_results):
            print(name, ': ', value)
        print()
        y_pred = (test_predictions_weighted > 0.5)
        cm = confusion_matrix(test_labels, y_pred)
        print(cm)

    if oversample_data == 1:

        pos_features = train_features[bool_train_labels]
        neg_features = train_features[~bool_train_labels]
        pos_labels = train_labels[bool_train_labels]
        neg_labels = train_labels[~bool_train_labels]
        print(pos_features.shape, 'pos feature shape')


        ids = np.arange(len(pos_features))
        choices = np.random.choice(ids, len(neg_features))

        res_pos_features = pos_features[choices]
        res_pos_labels = pos_labels[choices]
        print(res_pos_features.shape, 'resampled pos features shape')

        resampled_features = np.concatenate([res_pos_features, neg_features], axis=0)
        resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis=0)

        order = np.arange(len(resampled_labels))
        np.random.shuffle(order)
        resampled_features = resampled_features[order]
        resampled_labels = resampled_labels[order]
        print(resampled_features.shape, 'resempled features shape')
        BUFFER_SIZE = 100000

        def make_ds(features, labels):
            ds = tf.data.Dataset.from_tensor_slices((features, labels))  # .cache()
            ds = ds.shuffle(BUFFER_SIZE).repeat()
            return ds

        pos_ds = make_ds(pos_features, pos_labels)
        neg_ds = make_ds(neg_features, neg_labels)
        for features, label in pos_ds.take(1):
            print("Features:\n", features.numpy())
            print()
            print("Label: ", label.numpy())

        resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])
        resampled_ds = resampled_ds.batch(BATCH_SIZE).prefetch(2)

        for features, label in resampled_ds.take(1):
            print(label.numpy().mean())


        resampled_steps_per_epoch = np.ceil(2.0 * neg / BATCH_SIZE)
        resampled_steps_per_epoch

        resampled_model = makeModel(resampled_features)
        resampled_model.load_weights(initial_weights)

        # Reset the bias to zero, since this dataset is balanced.
        output_layer = resampled_model.layers[-1]
        output_layer.bias.assign([0])

        val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_labels)).cache()
        val_ds = val_ds.batch(BATCH_SIZE).prefetch(2)

        resampled_history = resampled_model.fit(
            resampled_ds,
            epochs=EPOCHS,
            steps_per_epoch=resampled_steps_per_epoch,
            callbacks=[early_stopping],
            validation_data=val_ds)
        test_predictions_resempled = resampled_model.predict(test_features, batch_size=BATCH_SIZE)
        resempled_results = resampled_model.evaluate(test_features, test_labels,
                                                   batch_size=BATCH_SIZE, verbose=0)
        for name, value in zip(resampled_model.metrics_names, resempled_results):
            print(name, ': ', value)
        print()
        y_pred = (test_predictions_resempled > 0.5)
        cm = confusion_matrix(test_labels, y_pred)
        print(cm)


    return



runKeras(1)

