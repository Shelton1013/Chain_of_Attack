Generate caption for a given image using [Unidiffuser](https://github.com/thu-ml/unidiffuser):

```
python i2t_unidiffuser.py
    --config "Unidiffuser/configs/sample_unidiffuser_v1.py" \
    --nnet_path "Unidiffuser/models/uvit_v1.pth" \
    --output_path [path_to_save_your_generated_caption] \
    --prompt "an elephant under the sea" \
    --n_samples 1000 \
    --nrow 4 \
    --mode i2t \
```