
  
import tensorflow as tf

class DataLoader():
    def __init__(self, csv_file='data/nyu2_train.csv', DEBUG=False):
       
        self.shape_depth = (240, 320, 1)
        self.read_nyu_data(csv_file, DEBUG=DEBUG)

    def read_nyu_data(self, csv_file, DEBUG=False):
        csv = open(csv_file, 'r').read()
        nyu2_train = list((row.split(',') for row in (csv).split('\n') if len(row) > 0))

        # Test on a smaller dataset
        if DEBUG: nyu2_train = nyu2_train[:10]    

        # A vector of depth filenames.
        self.labels = [i[1] for i in nyu2_train]

        # Length of dataset
        self.length = len(self.labels)
        print(self.length)
    def _parse_function(self, x_train): 
        # Read images from disk
        depth_resized = tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(x_train)), [self.shape_depth[0], self.shape_depth[1]])

        # Format
        depth = tf.image.convert_image_dtype(depth_resized / 255.0, dtype=tf.float32)
        
        # Normalize the depth values (in m)
        depth = tf.clip_by_value(depth * 10, 0, 10)

        return depth, depth

    def get_batched_dataset(self, batch_size):
        self.dataset = tf.data.Dataset.from_tensor_slices((self.labels))
        self.dataset = self.dataset.shuffle(buffer_size=len(self.labels), reshuffle_each_iteration=True)
        self.dataset = self.dataset.repeat()
        self.dataset = self.dataset.map(map_func=self._parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        self.dataset = self.dataset.batch(batch_size=batch_size).prefetch(tf.data.AUTOTUNE)

        return self.dataset