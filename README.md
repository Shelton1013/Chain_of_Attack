# Chain of Attack: On the Robustness of Vision-Language Models Against Transfer-Based Adversarial Attacks

Most existing transfer-based attacks neglect the importance of the semantic correlations between vision and text modalities, leading to sub-optimal adversarial example generation and attack performance. To address this issue, we present **Chain of Attack (CoA)**, which iteratively enhances the generation of adversarial examples based on the multi-modal semantic update using a series of intermediate attacking steps, achieving superior adversarial transferability and efficiency. A unified attack success rate computing method is further proposed for automatic evasion evaluation. Extensive experiments conducted under the most realistic and high-stakes scenario, demonstrate that our attacking strategy can effectively mislead models to generate targeted responses using only black-box attacks without any knowledge of the victim models. The comprehensive robustness evaluation in our paper provides insight into the vulnerabilities of VLMs and offers a reference for the safety considerations of future model developments. [[Paper](https://arxiv.org/pdf/2411.15720)]


## 🚀 News
- [28/02/2025] **Chain of Attack** is accepted by CVPR 2025!
- [24/11/2024] We introduce **Chain of Attack**, a new and efficient transfer-based attacking strategy for VLMs. The manuscript can be found on [arXiv](https://arxiv.org/pdf/2411.15720).

## Targeted image generation
Target caption: you can use the full targeted captions from MS-COCO
Then use [Stable Diffusion](https://github.com/CompVis/stable-diffusion), DALL-E or Midjourney to generate targeted images. 

## Chain of Attack
In our COA process, we need to use [Clipcap](https://github.com/rmokady/CLIP_prefix_caption)(you may access the pre-trained model weights and refer to the inference code available at this repository for further implementation details).
```
python train.py \
        --batch_size 1 \
        --num_samples 10000 \
        --input_res 224\
        --clip_encoder ViT-B/32 \
        --alpha 1.0 \
        --p_neg 0.7 \
        --epsilon 8 \
        --pgd_steps 100 \
        --a_weight 0.3 \
        --speed_up False\
        --update_steps 1 \
        --output "/COA/your_output_folder"\
        --cle_data_path "/COA/datasets/dataset_name"\
        --tgt_data_path "/COA/generated_targeted_images" \
        --model_path "/COA/clip_prefix_model/conceptual_weights.pt" \
        --prefix_length 10 \
        --tgt_file_path "/COA/coco_captions_1000.txt" \
        --cle_file_path "/COA/img_caption/llava_textvqa.txt" \
        --fusion_type "add_weight \
```

## 🔍Evaluation
For evaluation, we provide two metrics:

CLIP Score – You can compute this metric using the following script.

Attack Success Rate (ASR) – This metric can be evaluated using GPT-4 with our provided prompt.
```
python eval.py \
    --batch_size 100\
    --num_samples 10000\
    --pred_text_path "generated image caption" \
    --tgt_text_path  "target caption"\
```
  
## 📚Citation
```bibtex
@article{xie2024chain,
  title={Chain of Attack: On the Robustness of Vision-Language Models Against Transfer-Based Adversarial Attacks},
  author={Xie, Peng and Bie, Yequan and Mao, Jianda and Song, Yangqiu and Wang, Yang and Chen, Hao and Chen, Kani},
  journal={arXiv preprint arXiv:2411.15720},
  year={2024}
}
```

## 📄License

## 🙏Acknowledgement
