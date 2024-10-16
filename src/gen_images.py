'''
This is a demo code to generate image dataset
Specific usage can be adjusted according to needs.
'''

import os
import pickle
from diffusers import DiffusionPipeline, UNet2DConditionModel
import torch
from PIL import Image
import io
from tqdm import tqdm
import json

# width = 512 #@param {type: "number"}
# height = 640 #@param {type: "number"}
# steps = 50  #@param {type:"slider", min:1, max:50, step:1}
cfg_scale = 7.5 #@param {type:"slider", min:1, max:16, step:0.5}
# sample_cnt = 8 #@param {type:"number"}

def generate_images(pipeline, text, num_images=30):
    data = []
    for _ in range(num_images):
        image = pipeline(prompt=text, negative_prompt=negative_prompt,guidance_scale=cfg_scale).images[0]
        # image = pipeline(prompt=text).images[0]
        data.append({"image": image, "text": text})
    return data

def save_images(data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    metadata = []
    for idx, item in tqdm(enumerate(data), desc="Saving images"):
        image = item["image"]
        text = item["text"]
        image_filename = f"{idx+1:04d}.png"
        image_path = os.path.join(output_dir, image_filename)
        image.save(image_path)
        metadata.append({"file_name": image_filename, "text": text})
    metadata_file_path = os.path.join(output_dir, "metadata.jsonl")
    with open(metadata_file_path, "w") as f:
        for entry in metadata:
            json.dump(entry, f)
            f.write("\n")
    print(f"Images and metadata saved successfully in {output_dir}.")

negative_prompt = "bad anatomy,watermark,extra digit,signature,worst quality,jpeg artifacts,normal quality,low quality,lowres,error,blurry,text,cropped,username"
# negative_prompt = "bad anatomy,watermark,extra digit,signature,worst quality,jpeg artifacts,normal quality,low quality,long neck,lowres,error,blurry,missing fingers,fewer digits,missing arms,text,cropped,Humpbacked,bad hands,username"

model_path = "CompVis/stable-diffusion-v1-4" # "black-forest-labs/FLUX.1-dev"
root_dir = "./dataset/"
data_dir = "./data/coco_object_retain.csv"

import pandas as pd
data_promt = pd.read_csv(data_dir)
text_all_p = data_promt["prompt"].values
print(len(text_all_p))


text_dict = {"woman":["woman"],
            "man":["a photo of a man"],
             "hrm":["nudity"],
             "norm":["a lovely dog"],
             "style":["Thomas Kinkade"],
             "paint":["a painting"],
             "ft_style":["a big garden by Thomas Kinkade", "a village in the forest by Thomas Kinkade", "a peaceful park by Thomas Kinkade", "a beautiful house by Thomas Kinkade", "a colorful tree by Thomas Kinkade"],
             }


for type_name in ["style","paint"]: #["ft_style"]: #["hrm","rel"]:
    
    output_dir = root_dir+type_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create output directories
    train_image_dir = os.path.join(output_dir, "train")
    test_image_dir = os.path.join(output_dir, "test")
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(test_image_dir, exist_ok=True)

    pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipeline.safety_checker = None
    # pipeline.set_progress_bar_config(disable=True)
    pipeline.to("cuda")

    text_all = text_dict[type_name]

    train_data = []
    test_data = []
    for text in text_all:
        train_data.extend(generate_images(pipeline, text, num_images=30))
        test_data.extend(generate_images(pipeline, text, num_images=30))


    save_images(train_data, train_image_dir)
    save_images(test_data, test_image_dir)
