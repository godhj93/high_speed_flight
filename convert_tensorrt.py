import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import argparse
from tqdm import tqdm
from utils.data import data_load


parser = argparse.ArgumentParser(description='Convert tensorflow model to TensorRT model')

parser.add_argument('--load_model', required=True, help='path tensorflow model(saved format)')
parser.add_argument('--save_model', required=False, default='model.rt', help='path to save tensorRT model')
parser.add_argument('--quantize',required=False, default='FP16', help='[fp32,fp16,int8]')

args = parser.parse_args()


train_ds = data_load(batch_size=1)
train_ds = train_ds.get_batched_dataset()

def my_input_fn():
    for i,(x,y) in tqdm(enumerate(train_ds)):

        yield [x]
        if i == 500:
            break

if args.quantize== "FP32" or args.quantize== "fp32":
    print("\n\nFP32\n\n")
    conversion_params = trt.TrtConversionParams( precision_mode = trt.TrtPrecisionMode.FP32)

elif args.quantize== "FP16" or args.quantize== "fp16":
    print("\n\nFP16\n\n")
    conversion_params = trt.TrtConversionParams( precision_mode = trt.TrtPrecisionMode.FP16)


elif args.quantize== "int8" or args.quantize== "INT8":
    print("\n\nINT8\n\n")
    conversion_params = trt.TrtConversionParams( precision_mode = trt.TrtPrecisionMode.INT8)
    


print('converting....')
converter = trt.TrtGraphConverterV2(input_saved_model_dir=args.load_model, conversion_params = conversion_params)
if args.quantize == "int8" or args.quantize== "INT8":
    converter.convert(calibration_input_fn=my_input_fn)
else:
    converter.convert()
converter.build(input_fn=my_input_fn)
print('Done!')

save_path = './' + args.load_model + '_rt'
converter.save(save_path)
print('saved tensorRT model to {}'.format(save_path))