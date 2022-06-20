  
import tensorflow as tf

class data_load():
            
    def __init__(self, csv_file='data/noisy_train_ds.csv', batch_size=32,size= 256, DEBUG=False):
       
        self.shape_depth = (size, size, 1)
        self.read_nyu_data(csv_file, DEBUG=DEBUG)
        self.batch_size = batch_size


    def read_nyu_data(self, csv_file, DEBUG=False):
        csv = open(csv_file, 'r').read()
        nyu2_train = list((row.split(',') for row in (csv).split('\n') if len(row) > 0))

        # Test on a smaller dataset
        if DEBUG: nyu2_train = nyu2_train[:100]    

        self.images = [i[0] for i in nyu2_train]
        # A vector of depth filenames.
        self.labels = [i[1] for i in nyu2_train]

        # Length of dataset
        self.length = len(self.labels)
        print(f"Data length:{self.length}")
    def _parse_function(self, noisy, pure): 
        # Read images from disk
        noisy_img = tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(noisy)), [self.shape_depth[0], self.shape_depth[1]])
        pure_img = tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(pure)), [self.shape_depth[0], self.shape_depth[1]])

        # Normalization
        noisy_img = tf.image.convert_image_dtype(noisy_img / 255.0, dtype=tf.float32)
        pure_img = tf.image.convert_image_dtype(pure_img / 255.0, dtype=tf.float32)
        
        # Normalize the depth values (in m)
        # pure_img = tf.clip_by_value(pure_img * 10, 0, 10)

        #noisy_img = noisy_img * (tf.math.reduce_max(pure_img)/ tf.math.reduce_max(noisy_img))
        
        return noisy_img, pure_img

    def get_batched_dataset(self):
        self.dataset = tf.data.Dataset.from_tensor_slices((self.images, self.labels))
        self.dataset = self.dataset.shuffle(buffer_size=len(self.labels), reshuffle_each_iteration=True)
        self.dataset = self.dataset.repeat()
        self.dataset = self.dataset.map(map_func=self._parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        self.dataset = self.dataset.batch(batch_size=self.batch_size).prefetch(tf.data.AUTOTUNE)

        return self.dataset