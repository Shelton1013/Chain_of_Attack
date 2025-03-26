we provide the code to use Unidiffuser to generate caption, for more detail please refer to [Unidiffuser](https://github.com/thu-ml/unidiffuser)

```
python i2t.py
    --config "/Unidiffuser/configs/sample_unidiffuser_v1.py" \
    --nnet_path "/Unidiffuser/models/uvit_v1.pth" \
    --output_path "path to save your generated caption" \
    --prompt "an elephant under the sea" \
    --n_samples 1000 \
    --nrow 4 \
    --mode "i2t" \
```