#!/usr/bin/env python3.8
import tensorflow as tf
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import time
from cv_bridge import CvBridge, CvBridgeError
import argparse

parser = argparse.ArgumentParser('High Speed Flight')
parser.add_argument("--model", type=str, help= "Model Path", default='./models/Encoder_RELU_256_64')
parser.add_argument("--input_size", type=int, help= "Input dimension", default=256)
args = parser.parse_args()

#Limit GPU Memory Usage 
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)


class Depth_extractor:
    '''
    ROS Node Name: /depth_states
    '''
    def __init__(self, model_path, input_dim=256):

        '''
        model_path: Path to Load a Model
        input_dim: Size of Input Image
        '''

        self.input_dim = input_dim

        #Loading the Neural Network(Tensor RT)
        print("Loading the model...")
        self.model = tf.saved_model.load(model_path+'/tensor_rt')
        print("Done.")

        #Preparing Inference for Tensor RT
        self.infer = self.model.signatures['serving_default']
        self.output_tensorname = list(self.infer.structured_outputs.keys())[0]
        # print(self.model.summary())
        
        print("Initilizing ROS...")
        #Subscribing Depth Image from the Simulator
        rospy.Subscriber("/depth", Image, self.depth_cb)
        self.depth_bridge = CvBridge()
        #Publishing States for the Agent
        self.depth_state_pub = rospy.Publisher("/depth_states", Float32MultiArray, queue_size=10)
        self.depth_states = Float32MultiArray()
        print("Done.")

    def inference(self, depth_img):
        
        # Resizing input image
        depth_img_resized = tf.image.resize(images = tf.expand_dims(depth_img, axis=-1), size = (self.input_dim, self.input_dim))
        
        # Normalize the Depth Value (in m)
        x = tf.image.convert_image_dtype(depth_img_resized / 255.0, dtype=tf.float32)
        x = tf.clip_by_value(x * 10, 0, 10)
        
        # Converting the Input to Batch 1: (256,256,1) -> (1,256,256,1)
        x = tf.cast(tf.expand_dims(x, axis=0), dtype=tf.float32)
        
        # Inference
        y_hat = self.infer(tf.constant(x, dtype=tf.float32))[self.output_tensorname]
        
        return y_hat[0].numpy().tolist()

    def depth_cb(self, data):
        
        try:
            t0 = time.time()

            depth = self.depth_bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            
            self.depth_states.data = self.inference(depth)
            self.depth_state_pub.publish(self.depth_states)

            t1 = time.time()
            print(f"Latency: {(t1-t0)*1000:.4f}ms")

            return 0

        except CvBridgeError as e:
            print(e)


if __name__ == '__main__':

    rospy.init_node('depth_extractor', anonymous=True)
    
    depth_extractor = Depth_extractor(model_path=args.model, input_dim=args.input_size)

    rate = rospy.Rate(100)

    while not rospy.is_shutdown():

        rate.sleep()

        


