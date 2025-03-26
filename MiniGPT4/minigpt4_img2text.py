import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse

import random
from PIL import Image
import time

import numpy as np
import torch
import torchvision
import torch.backends.cudnn as cudnn

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


# seed for everything
# credit: https://www.kaggle.com/code/rhythmcam/random-seed-everything
DEFAULT_RANDOM_SEED = 2023
device = "cuda" if torch.cuda.is_available() else "cpu"

# basic random seed
def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# torch random seed
def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# combine
def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)
# ------------------------------------------------------------------ #  


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        original_tuple = super().__getitem__(index)  # (img, label)
        path, _ = self.samples[index]  # path: str

        image_processed = vis_processor(original_tuple[0])
        return image_processed, original_tuple[1], path
    
    
if __name__ == "__main__":
    seedEverything()
    parser = argparse.ArgumentParser(description="Demo")
    
    # minigpt-4
    parser.add_argument("--cfg-path", default="/home/pxieaf/AttackVLM/MiniGPT-4/MiniGPT-4/eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    
    # obtain text in batch
    parser.add_argument("--query", default='describe this image in one sentence.', type=str)

    parser.add_argument("--dataset_path", default="your generated image dataset path", type=str)
    parser.add_argument("--save_path", default="save caption path", type=str)

    args = parser.parse_args()
    
    conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0}
    
    print(f"Loading MiniGPT-4 model...")

    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)  # model_config.arch: minigpt-4
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

    CONV_VISION = conv_dict[model_config.model_type]

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor     = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)       
    num_beams = 1
    temperature = 1.0
    print("Done.")

    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))     
    
    directory = args.dataset_path

    image_folder = os.listdir(directory)
    image_folders = sorted(image_folder)

    for folder in image_folders:
        class_folder = directory + "/" +folder
        entries = os.listdir(class_folder)
        entries = sorted(entries) 

        for image in entries:
            image = class_folder + '/' +image

            with torch.no_grad():
                chat_state = CONV_VISION.copy()
                chat_state.append_message(chat_state.roles[0], "<Img><ImageHere></Img>")
                raw_image = Image.open(image).convert("RGB")
                img_list = []
                img_list.append(raw_image)
                chat.encode_img(img_list)

                chat.ask(args.query, chat_state)

                llm_message = chat.answer(conv=chat_state,
                                img_list=img_list,
                                num_beams=num_beams,
                                temperature=temperature,
                                max_new_tokens=300,
                                max_length=2000)[0]
                

            # write captions
            path = args.save_path
            with open(path, 'a') as f:    
                print(llm_message)
                f.write(llm_message + '\n')
      
    print("Caption generated successfully.")