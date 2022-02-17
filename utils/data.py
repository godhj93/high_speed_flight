import tensorflow as tf
from tensorflow.keras.datasets import cifar10

tf.random.set_seed(777)

def data_load(batch_size= 16, size= 256, DEBUG=False):
    
    AUTO = tf.data.AUTOTUNE
    (x_train,y_train), (x_test, y_test) = cifar10.load_data()

    if DEBUG==True:
        print(f"Debug Mode")
        (x_train,y_train) = (x_train[:1000,:,:,:], y_train[:1000,:])
        (x_test,y_test) = (x_test[:100,:,:,:], y_test[:100,:])
        print(f"length of train and test data: {len(x_train), len(x_test)}")

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train,y_train)).
        batch(batch_size).
        shuffle(batch_size*100, seed=777, reshuffle_each_iteration=True).
        map(resize_and_normalization)
        ).prefetch(AUTO)

    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test,y_test)).
        batch(batch_size).
        map(resize_and_normalization)        
    ).prefetch(AUTO)

    return train_ds, test_ds

def resize_and_normalization(x,y):

    x = tf.image.resize(x, size=(256,256), method='bicubic')
    x = tf.cast(x/255.0, dtype=tf.float32)

    return x,x

    
