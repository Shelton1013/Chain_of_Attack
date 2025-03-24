import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import random
import clip
import numpy as np
import torch
import torchvision
from PIL import Image
import torch.nn as nn


import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from PIL import Image
import time

# seed for everything
# credit: https://www.kaggle.com/code/rhythmcam/random-seed-everything
DEFAULT_RANDOM_SEED = 2024
device = "cuda" if torch.cuda.is_available() else "cpu"

import torchvision.transforms as T
to_pil = T.ToPILImage()


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

def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        original_tuple = super().__getitem__(index)
        path, _ = self.samples[index]
        return original_tuple + (path,)
    

def custom_data_loader(file_path, batch_size):
    with open(file_path, 'r', encoding='utf-8') as file:
        batch = []
        for line in file:
            path = line.strip() 
            batch.append(path)
            
            if len(batch) == batch_size:
                yield batch  
                batch = []  

        if batch:
            yield batch

# def tensor_to_pil(image_tensor):
#     # Convert torch.Tensor back to PIL.Image
#     if image_tensor.dim() == 4:
#         image_tensor = image_tensor[0] 
#     image_tensor = image_tensor.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
#     return Image.fromarray(image_tensor)

# def tensor_to_pil(image_tensor):
#     image_np = image_tensor.permute(1, 2, 0).detach().cpu().numpy() 
#     image_np = (image_np * 255).astype('uint8') 
    
#     image_pil = Image.fromarray(image_np)
#     return image_pil

##################
# clip prefix 
N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]

D = torch.device
CPU = torch.device('cpu')


class MLP(nn.Module):

    def forward(self, x: T) -> T:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) -1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class ClipCaptionModel(nn.Module):

    #@functools.lru_cache #FIXME
    def get_dummy_token(self, batch_size: int, device: D) -> T:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        #print(embedding_text.size()) #torch.Size([5, 67, 768])
        #print(prefix_projections.size()) #torch.Size([5, 1, 768])
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:  # not enough memory
            self.clip_project = nn.Linear(prefix_size, self.gpt_embedding_size * prefix_length)
        else:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2, self.gpt_embedding_size * prefix_length))


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self
    

def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                  entry_length=67, temperature=1., stop_token: str = '.'):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def generate_cap(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in trange(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]



def train():
    args = parser.parse_args()
    
    alpha = args.alpha
    epsilon = args.epsilon
    p_neg = args.p_neg
    fusion_type = args.fusion_type
    speed_up = args.speed_up

    
    # load clip_model params
    clip_model, preprocess = clip.load(args.clip_encoder, device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # caption model init
    prefix_length = args.prefix_length
    cap_model = ClipCaptionModel(prefix_length)
    cap_model.load_state_dict(torch.load(args.model_path, map_location=CPU)) 
    cap_model = cap_model.eval() 
    cap_model = cap_model.to(device)

    # ------------- pre-processing images/text ------------- #
    
    # preprocess images
    transform_fn = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(args.input_res, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(args.input_res),
            torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
            torchvision.transforms.Lambda(lambda img: to_tensor(img)),
        ]
    )
    clean_data    = ImageFolderWithPaths(args.cle_data_path, transform=transform_fn)
    target_data   = ImageFolderWithPaths(args.tgt_data_path, transform=transform_fn)
    
    data_loader_imagenet = torch.utils.data.DataLoader(clean_data, batch_size=args.batch_size, shuffle=False, num_workers=12, drop_last=False)
    data_loader_target   = torch.utils.data.DataLoader(target_data, batch_size=args.batch_size, shuffle=False, num_workers=12, drop_last=False)
    

    clip_preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(clip_model.visual.input_resolution, interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True),
            torchvision.transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            torchvision.transforms.CenterCrop(clip_model.visual.input_resolution),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # CLIP imgs mean and std.
        ]
    )
    
    # CLIP imgs mean and std.
    inverse_normalize = torchvision.transforms.Normalize(mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711], std=[1.0 / 0.26862954, 1.0 / 0.26130258, 1.0 / 0.27577711])


    tgt_text_loader = custom_data_loader(args.tgt_file_path, args.batch_size)
    cle_text_loader = custom_data_loader(args.cle_file_path, args.batch_size)


    # training
    for i, ((image_org, _, path), (image_tgt, _, _), tgt_text, cle_text) in enumerate(zip(data_loader_imagenet, data_loader_target, tgt_text_loader, cle_text_loader)):
        if args.batch_size * (i+1) > args.num_samples:
            break
        
        # (bs, c, h, w)
        image_org = image_org.to(device) # (2,3,224,224)
        image_tgt = image_tgt.to(device)

        # add text
        cle_text = clip.tokenize(cle_text).to(device)
        tgt_text = clip.tokenize(tgt_text).to(device)
        

        # get tgt featutres
        with torch.no_grad():
            tgt_image_features = clip_model.encode_image(clip_preprocess(image_tgt)) # (b,512)
            tgt_image_features = tgt_image_features / tgt_image_features.norm(dim=1, keepdim=True)

            tgt_text_features = clip_model.encode_text(tgt_text)
            tgt_text_features = tgt_text_features / tgt_text_features.norm(dim=1, keepdim=True)

            cle_image_features = clip_model.encode_image(clip_preprocess(image_org)) # (b,512)
            cle_image_features = cle_image_features / cle_image_features.norm(dim=1, keepdim=True)

            cle_text_features = clip_model.encode_text(cle_text)
            cle_text_features = cle_text_features / cle_text_features.norm(dim=1, keepdim=True)
            
            # choose embedding fusion type
            if fusion_type == "cat":
                # cat
                cle_fused_embedding = torch.cat((cle_image_features, cle_text_features), dim=1)
                cle_fused_embedding = cle_fused_embedding / cle_fused_embedding.norm(dim=1, keepdim=True)
                tgt_fused_embedding = torch.cat((tgt_image_features, tgt_text_features), dim=1)
                tgt_fused_embedding = tgt_fused_embedding / tgt_fused_embedding.norm(dim=1, keepdim=True)
            elif fusion_type == "add_weight":
                # add with weight
                a = args.a_weight
                cle_fused_embedding = a * cle_image_features + (1-a) * cle_text_features
                cle_fused_embedding = cle_fused_embedding / cle_fused_embedding.norm(dim=1, keepdim=True)
                tgt_fused_embedding = a * tgt_image_features + (1-a) * tgt_text_features
                tgt_fused_embedding = tgt_fused_embedding / tgt_fused_embedding.norm(dim=1, keepdim=True)
            elif fusion_type == "multiplication":
                # multiplication
                tgt_fused_embedding = tgt_image_features * tgt_text_features
                tgt_fused_embedding = tgt_fused_embedding / tgt_fused_embedding.norm(dim=1, keepdim=True)
                cle_fused_embedding = cle_image_features * cle_text_features
                cle_fused_embedding = cle_fused_embedding / cle_fused_embedding.norm(dim=1, keepdim=True)
            else:
                raise ValueError(f"Unsupported fusion_type: {fusion_type}")

                
        # -------- get adv image -------- #
        delta = torch.zeros_like(image_org, requires_grad=True)
        total_time = 0
        for j in range(args.pgd_steps):
            start_time = time.time()
            adv_image = image_org + delta
            
            adv_image_clone = adv_image.clone()
            adv_image_clone = adv_image_clone / 255.0

            adv_image = clip_preprocess(adv_image)
            
            if speed_up == True:
                # to speed up the process, you can update the current caption every 2 steps
                if j % args.update_steps == 0:    
                    bs = adv_image_clone.size(0)  
                    current_caption_list = []
                    for k in range(bs):
                        image_pil = to_pil(adv_image_clone[k])
                        processed_image = preprocess(image_pil).unsqueeze(0).to(device)
                    
                        with torch.no_grad():
                            prefix = clip_model.encode_image(processed_image).to(device, dtype=torch.float32)
                            prefix_embed = cap_model.clip_project(prefix).reshape(1, prefix_length, -1)

                        current_caption = generate_cap(cap_model, tokenizer, embed=prefix_embed)
                        current_caption_list.append(current_caption)
                else:
                    current_caption_list = last_caption_list

                last_caption_list = current_caption_list

            else:
                bs = adv_image_clone.size(0)  
                current_caption_list = []
                for k in range(bs):
                    image_pil = to_pil(adv_image_clone[k])
                    processed_image = preprocess(image_pil).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        prefix = clip_model.encode_image(processed_image).to(device, dtype=torch.float32)
                        prefix_embed = cap_model.clip_project(prefix).reshape(1, prefix_length, -1)

                    current_caption = generate_cap(cap_model, tokenizer, embed=prefix_embed)
                    current_caption_list.append(current_caption)


            adv_image_features = clip_model.encode_image(adv_image) # (b,512)
            adv_image_features = adv_image_features / adv_image_features.norm(dim=1, keepdim=True)

            # current caption
            cur_caption = clip.tokenize(current_caption_list).to(device)
            cur_text_features = clip_model.encode_text(cur_caption)
            cur_text_features = cur_text_features / cur_text_features.norm(dim=1, keepdim=True)

            # choose embedding fusion type
            if fusion_type == "cat":
                # cat    
                cur_fused_embedding = torch.cat((adv_image_features, cur_text_features), dim=1)
                cur_fused_embedding = cur_fused_embedding / cur_fused_embedding.norm(dim=1, keepdim=True)
            elif fusion_type == "add_weight":    
                # add with weight
                cur_fused_embedding = a * adv_image_features + (1-a) * cur_text_features 
                cur_fused_embedding = cur_fused_embedding / cur_fused_embedding.norm(dim=1, keepdim=True)
            elif fusion_type == "multiplication":
                # multiplication
                cur_fused_embedding = adv_image_features * cur_text_features
                cur_fused_embedding = cur_fused_embedding / cur_fused_embedding.norm(dim=1, keepdim=True)
            else:
                raise ValueError(f"Unsupported fusion_type: {fusion_type}")


            embedding_sim1 = torch.mean(torch.sum(cur_fused_embedding * cle_fused_embedding, dim=1)) # pos
            embedding_sim2 = torch.mean(torch.sum(cur_fused_embedding * tgt_fused_embedding, dim=1)) # neg
            
            """
            # triplet loss, origin triplet loss use l2 dis, we use cos sim
            # anchor = cur_fused_embedding
            # positive = tgt_fused_embedding
            # negative = cle_fused_embedding
            # loss = triplet_loss(anchor, positive, negative)
            """

            margin = 1 - p_neg
            loss = torch.mean(torch.relu(embedding_sim2 - p_neg * embedding_sim1 + margin))
            loss.backward()
            # embedding_sim = torch.mean(torch.sum(adv_image_features * tgt_image_features, dim=1))  # computed from normalized features (therefore it is cos sim.)
            # embedding_sim.backward()
            
            grad = delta.grad.detach()
            d = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
            delta.data = d
            delta.grad.zero_()

        """
        # compute cost time
            end_time = time.time()
            one_step_time = end_time - start_time
            total_time += one_step_time
            print("one step time cost:", one_step_time)
            print(f"iter {i}/{args.num_samples//args.batch_size} step:{j:3d}, cos sim={embedding_sim2.item():.5f}, max delta={torch.max(torch.abs(d)).item():.3f}, mean delta={torch.mean(torch.abs(d)).item():.3f}")
        
        print("total time", total_time/args.pgd_steps)
        """
        
        # save imgs
        adv_image = image_org + delta
        adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)
        for path_idx in range(len(path)):
            folder, name = path[path_idx].split("/")[-2], path[path_idx].split("/")[-1]
            folder_to_save = os.path.join(args.output, folder)
            if not os.path.exists(folder_to_save):
                os.makedirs(folder_to_save, exist_ok=True)
            if 'JPEG' in name:
                torchvision.utils.save_image(adv_image[path_idx], os.path.join(folder_to_save, name[:-4]) + 'png')
            elif 'jpg' in name:
                torchvision.utils.save_image(adv_image[path_idx], os.path.join(folder_to_save, name))
        


if __name__ == "__main__":
    seedEverything()
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_samples", default=100, type=int)
    parser.add_argument("--input_res", default=224, type=int)
    parser.add_argument("--clip_encoder", default="ViT-B/32", type=str)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--p_neg", default=0.7, type=float)
    parser.add_argument("--epsilon", default=8, type=int)
    parser.add_argument("--pgd_steps", default=100, type=int)
    parser.add_argument("--a_weight", default=0.3, type=float,help='embedding fusion type add_weight, weight')
    parser.add_argument("--speed_up", default=False, type=bool, help='speed up chain of attack')
    parser.add_argument("--update_steps", default=1, type=int, help='chain of attack speed up step')
    parser.add_argument("--output", default="/COA/your_output_folder", type=str, help='the folder name of output')
    
    parser.add_argument("--cle_data_path", default="/COA/datasets/dataset_name", type=str, help='path of the clean images')
    parser.add_argument("--tgt_data_path", default="/COA/generated_targeted_images", type=str, help='path of the target images')
    parser.add_argument("--model_path", default="/COA/clip_prefix_model/conceptual_weights.pt", type=str, help='path of the image2text model')
    parser.add_argument("--prefix_length", default=10, type=int, help='image2text model prefix length')
    
    
    parser.add_argument("--tgt_file_path", default="/COA/coco_captions_1000.txt", type=str, help='path of the target text')
    parser.add_argument("--cle_file_path", default="/COA/img_caption/llava_textvqa.txt", type=str, help='path of the clean image text')

    parser.add_argument("--fusion_type", default="add_weight", type=str, help='embedding fusion type: cat, add_weight, multiplication')
    
    train()
