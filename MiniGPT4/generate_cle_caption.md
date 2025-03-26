we provide the code to use MiniGPT-4 to generate clean caption, for more detail please refer to [MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4)

```
python minigpt4_img2text.py
    --cfg-path "minigpt4_eval.yaml" \
    --gpu-id 0 \
    --query 'describe this image in one sentence.' \
    --dataset_path "your generated image dataset path"\
    --save_path "save caption path" \
```