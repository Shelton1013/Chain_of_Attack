To generate clean texts (captions) for the clean images using [MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4), please use the following command: 

```
python minigpt4_img2text.py
    --cfg-path "minigpt4_eval.yaml" \
    --gpu-id 0 \
    --query "describe this image in one sentence." \
    --dataset_path [clean_image_path] \
    --save_path [caption_save_path]
```
