# train instruction
## prepare training data
After data adaptation, run `utils/merge_jsons.py` to combine training datasets as desired (m4avg/m4proportional/db4cn-wdb/db4en-wdb/db4cn-shift-wdb/db4en-shift-wdb/...) to generate the training data, as a `.json` file.
## training original DiffSinger model from original [M4Singer repo](https://github.com/M4Singer/M4Singer)
Almost follow [the instrcution from M4Singer](https://github.com/M4Singer/M4Singer/blob/master/code/README.md), with little modification to `M4Singer/code/data_gen/singing/binarize.py` to accomodate our structured data, new file is provided at `train_m4singer/binarize.py`.

### specific training pipeline (we follow [the instruction from M4Singer](https://github.com/M4Singer/M4Singer/blob/master/code/README.md))

- We use the pre-trained [Vocoder](https://drive.google.com/file/d/10LD3sq_zmAibl379yTW5M-LXy2l_xk6h/view?usp=share_link) and [PitchExtractor](https://drive.google.com/file/d/19QtXNeqUjY3AjvVycEt3G83lXn2HwbaJ/view?usp=share_link).
- We train FFT-Singer from scratch to get a pre-trained FFT-Singer checkpoint (320000 steps).
- for each experimen `exp_name`, under `code/usr/configs`, make a directory named as `exp_name`, like this:
```sh
- code/usr/configs
- code/usr/configs/exp_name
- code/usr/configs/exp_name/base.yaml
- code/usr/configs/exp_name/fs2.yaml
- code/usr/configs/exp_name/diff.yaml
```
The three `.yaml` file are based on [provided](https://github.com/M4Singer/M4Singer/tree/master/code/usr/configs/m4singer), customize these config files as follows:
- **IMPORTANT**: in `code/usr/configs/exp_name/base.yaml`, comment key `datasets` and add key `raw_json_fn`, the prepared structued json file, which will be read when run `train_m4singer/binarize.py` to binarize the training data.
- and normally set experimental paths:
    - in `code/usr/configs/exp_name/base.yaml`, set `data/binary/exp_name` as value for key `binary_data_dir`.
    - in `code/usr/configs/exp_name/fs2.yaml`, set `configs/singing/fs2.yaml` and `usr/configs/exp_name/base.yaml` as value for key `base_config`.
    - in `code/usr/configs/exp_name/diff.yaml`, set `usr/configs/exp_name/base.yaml` as value for key `base_config`; set `checkpoints/exp_name_fs2_e2e` as value for key `fs2_ckpt`.


## training our BiSinger model
Complete code for BiSinger are under `train_bisinger` directory. The training routine are similar to above.

## inference
We also provide our inference scripts for the five systems in [our paper](https://arxiv.org/abs/2309.14089), please refer to `train_m4singer/bisinger-inference` and `train_bisinger/inference/m4singer/bisinger`.

Under `train_bisinger/`, we run the following command for inference.
```sh
CUDA_VISIBLE_DEVICES=0 python inference/m4singer/bisinger/a-m4-detect.py --config usr/configs/m4-detect-ori-shift/diff.yaml --exp_name m4-detect-ori-shift_diff_e2e --infer
```







