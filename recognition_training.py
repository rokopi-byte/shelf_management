import argparse
import datetime
import os
from pathlib import Path
import tensorflow as tf
import tensorflow_addons as tfa


def read_image(filename, target_shape):
    """
    Preprocessing and image read
    :param filename: image file name
    :param target_shape: target shape for the model
    :return:tensorflow image object
    """
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    image = tf.keras.applications.mobilenet_v3.preprocess_input(image)
    return image


def train(target_shape, embedding_size, batch_size, num_epochs, data_dir):
    '''
    Main method for traning.
    :param target_shape: target shape for the model
    :param embedding_size: size of the embedding vector to generate
    :param batch_size: batch size
    :param num_epochs: number of epochs, early stopping is used, see code.
    :param data_dir: directory for the training set
    :return:
    '''
    base_cnn = tf.keras.applications.MobileNetV3Large(
        weights="imagenet", input_shape=target_shape + (3,), include_top=False
    )
    flatten = tf.keras.layers.Flatten()(base_cnn.output)
    dense1 = tf.keras.layers.Dense(embedding_size)(flatten)
    output = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(dense1)
    embeddingNet = tf.keras.Model(base_cnn.input, output, name="Embedding")

    data_dir = Path(data_dir, '**', '**', '*.jpg')
    list_ds = tf.data.Dataset.list_files(str(data_dir), shuffle=False)
    image_count = len(list_ds)
    label_list = []
    for i in (list_ds):
        label_list = label_list + [int(Path(i.numpy().decode()).parent.name)]

    label_list_tf = tf.data.Dataset.from_tensor_slices(label_list)
    dataset = list_ds.map(lambda x: read_image(x, target_shape))
    dataset = tf.data.Dataset.zip((dataset, label_list_tf))
    dataset = dataset.shuffle(buffer_size=1024)

    train_dataset = dataset.take(round(image_count * 0.8))
    val_dataset = dataset.skip(round(image_count * 0.8))

    train_dataset = train_dataset.batch(batch_size, drop_remainder=False)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    val_dataset = val_dataset.batch(batch_size, drop_remainder=False)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    logdir = os.path.join(".", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('model', monitor='val_loss', verbose=1,
                                                                   save_best_only=True, save_weights_only=False,
                                                                   mode='auto', save_freq='epoch')
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.001, patience=7,
                                                               verbose=1,
                                                               mode="auto", baseline=None, restore_best_weights=False)

    embeddingNet.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tfa.losses.TripletHardLoss(soft=True))

    history = embeddingNet.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=val_dataset,
        callbacks=[tensorboard_callback, model_checkpoint_callback, early_stopping_callback])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_shape', type=int, required=True)
    parser.add_argument('--embedding_size', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--num_epochs', type=int, required=True)
    parser.add_argument('--data_dir', type=str, required=True)

    args = parser.parse_args()

    target_shape = (args.target_shape, args.target_shape)
    embedding_size = args.embedding_size
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    data_dir = args.data_dir

    train(target_shape, embedding_size, batch_size, num_epochs, data_dir)
