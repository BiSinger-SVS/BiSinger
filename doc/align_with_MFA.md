# overview

We use Montreal Forced Aligner (MFA) to extract phoneme duration for DB-4 and M4Singer datasets. More details can refer to the [Montreal Forced Aligner tutorial by Eleanor Chodroff](https://lingmethodshub.github.io/content/tools/mfa/mfa-tutorial). Due to the provided pronounciation dictionary correspond to pretrained model for Chinese and English are based on different phoneme sets, we choose to train new acoustic models on DB-4 and M4Singer, respectively. According to [the official training tutorial](https://montreal-forced-aligner.readthedocs.io/en/latest/first_steps/index.html#first-steps-align-train-acoustic-model), there're some data needed:
- paired audio and transcript (orthographic annotations)
- the pronounciation dictionary

Seeking for a unified phoneme set, we adopt the CMU Pronunciation Dictionary, specificly [ the official CMU dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) for English and [the Pinyin-to-CMU mapping](https://github.com/kaldi-asr/kaldi/blob/master/egs/hkust/s5/conf/pinyin2cmu) for Chinese.

After obtaining the dictionary, we prepare the paired data for training the applicable acoustic model, and use it to align the audio and transcript, ultimately getting the phoneme duration.

## for DB-4

### read transcript from raw dateset

Run `utils/get_pairs_db4.sh` to read cleaned text transcript, getting `data/mfa_preparation/output_cn.txt` and `data/mfa_preparation/output_en.txt`.

```sh
# demo from data/mfa_preparation/output_cn.txt
000001|心 动 不 如 行 动 我 不 太 擅 长 卖 萌|xin dong bu ru xing dong wo bu tai shan chang mai meng
000002|君 子 动 口 不 动 手 哦 小 猫 得 替 我 逮 耗 子|jun zi dong kou bu dong shou o xiao mao dei ti wo dai hao zi
000003|我 回 右 哼 哼 左 哼 哼 云 峰 和 精 灵 同 居 了|wo hui you heng heng zuo heng heng yun feng he jing ling tong ju le
000004|谣 传 珍 与 有 妇 之 夫 鬼 混|yao chuan zhen yu you fu zhi fu gui hun

# demo from data/mfa_preparation/output_en.txt
300001|Hang on gaps those of you in the know|HH AE NG  AA N  G AE P Z  DH OW Z  AH V  Y UW  IH N  DH AH  N OW
300002|Stay cool he added with a smile|S T EY  K UW L  HH IY  AE  D IH D  W IH DH  AH  S M AY L
300003|When I found out about her death I was shock but not surprised she said|W EH N  AY  F AW N D  AW T  AH  B AW T  HH ER  D EH TH  AY  W AA Z  SH AA K  B AH T  N AA T  S ER  P R AY Z D  SH IY  S EH D
300004|his sister Sara asked|HH IH Z  S IH  S T ER  S EH  R AH  AE S K T
```

### pair input data as required by MFA

Make workspace directory for DB4-CN and DB4-EN, under each, make subdirectory `input` (contains input data to MFA) and `output` (an empty ).
Run `utils/get_input_db4.sh` to prepare paired input data: audio file (.wav) and transcript (.txt). It's expected to get:

```sh
- mfa_workspace
    - db4-cn
        - input
            - 000001.txt
            - 000001.wav
            - 000002.txt
            - 000002.wav
            - ...
        - output
            (empty)
    - db4-en
        - input
            - 300001.txt
            - 300001.wav
            - 300002.txt
            - 300002.wav
            - ...
        - output
            (empty)
# for transcript of db4-cn, it looks like:
    xin dong bu ru xing dong wo bu tai shan chang mai meng
# for transcript of db4-en, it looks like:
    Hang on gaps those of you in the know
```

### respectively train the acoustic model on DB4-CN and DB4-EN

```sh
# training on db4-cn
docker run --name mfacn -it -v data/mfa_workspace:/data mmcauliffe/montreal-forced-aligner:v2.2.10
mfa validate data/db4-cn/input data/dict/lexicon-cn.txt
mfa train data/db4-cn/input data/dict/lexicon-cn.txt data/acoustic_db4-cn.zip

# training on db4-en
docker run --name mfaen -it -v data/mfa_workspace:/data mmcauliffe/montreal-forced-aligner:v2.2.10
mfa validate data/db4-en/input data/dict/lexicon-en.txt
mfa train data/db4-en/input data/dict/lexicon-en.txt data/acoustic_db4-en.zip
```

### align the audio and transcript with the well-trained model
After training, we will get a brand new prounciation dictionary with probability (same file name with the provided one during training) and a .zip acoustic model. In this step, we use it to align audio and transcript at phoneme level, acquiring the phoneme duration.

**NOTE**: Be careful with the fact that there are `data/mfa_workspace/dict/lexicon-cn.txt` for training (obtained from open-source website mentioned above) and probability prounciation dictionary `data/mfa_workspace/lexicon-cn.dict` obtained from training.

```sh
docker run --name mfacn -it -v data/mfa_workspace:/data mmcauliffe/montreal-forced-aligner:v2.2.10
mfa align --clean data/db4-cn/input data/lexicon-cn.dict data/acoustic_db4-cn.zip output/ --beam 400 --retry_beam 1000

docker run --name mfaen -it -v data/mfa_workspace:/data mmcauliffe/montreal-forced-aligner:v2.2.10
mfa align --clean data/db4-en/input data/lexicon-en.dict data/acoustic_db4-en.zip output/ --beam 400 --retry_beam 1000
```

## for M4Singer

It's similar procedure to the DB-4: prepare paired data ==> train new acoustic model on it ==> align the audio and transcript with obtained probability prounciation dictionary and model.

### read transcript from raw dateset & pair input data as required by MFA

Run `utils/get_pairs_m4singer.py`.

### train the acoustic model on M4Singer

```sh
# training on M4Singer
docker run --name mfam4 -it -v data/mfa_workspace:/data mmcauliffe/montreal-forced-aligner:v2.2.10
mfa validate data/m4singer/input data/dict/lexicon-m4.txt
mfa train data/m4singer/input data/dict/lexicon-m4.txt data/acoustic_m4.zip
```

### align the audio and transcript with the well-trained model

```sh
mfa align --clean input/ ../lexicon-m4.dict ../acoustic_m4.zip output/ --beam 400 --retry_beam 1000
```
