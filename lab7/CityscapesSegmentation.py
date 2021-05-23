import os

from keras_segmentation.pretrained import pspnet_101_cityscapes

model = pspnet_101_cityscapes()  # load the pretrained model trained on Cityscapes dataset

input_path = "./cityscapes_data/input"
output_path = "./cityscapes_data/output"

for filename in os.listdir(input_path):
    if filename.endswith(".png"):
        out = model.predict_segmentation(
            inp=input_path+"/"+filename,
            out_fname=output_path+"/"+filename
        )
    else:
        continue

