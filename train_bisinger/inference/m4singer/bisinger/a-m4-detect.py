import glob
import json
import os
import re
from functools import reduce

import librosa
import numpy as np
import spacy
import torch
from inference.m4singer.m4singer.map import m4singer_pinyin2ph_func
from modules.fastspeech.pe import PitchExtractor
from modules.hifigan.hifigan import HifiGanGenerator
from pypinyin import Style, lazy_pinyin, pinyin
from spacy_syllables import SpacySyllables
from usr.diff.shallow_diffusion_tts import GaussianDiffusion
from usr.diffsinger_task import DIFF_DECODERS
from vocoders.hifigan import HifiGAN

import utils
from utils import load_ckpt
from utils.audio import save_wav
from utils.hparams import hparams, set_hparams
from utils.text_encoder import TokenTextEncoder

# ALL_YUNMU = ['a', 'ai', 'an', 'ang', 'ao',  'e', 'ei', 'en', 'eng', 'er',  'i', 'ia', 'ian', 'iang', 'iao',
#              'ie', 'in', 'ing', 'iong', 'iou', 'o', 'ong', 'ou', 'u', 'ua', 'uai', 'uan', 'uang', 'uei',
#              'uen', 'uo', 'v', 'van', 've', 'vn']


nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("syllables", after="tagger")
# print(nlp.pipe_names)
# assert nlp.pipe_names == ["tok2vec", "tagger", "syllables", "parser", "ner", "attribute_ruler", "lemmatizer"]
doc = nlp("terribly long")
data = [(token.text, token._.syllables, token._.syllables_count) for token in doc]
# print(data)
assert data == [("terribly", ["ter", "ri", "bly"], 3), ("long", ["long"], 1)]

CHINESE = 1
ENGLISH = 0


def b2s_in_a_word(bpm, beat_in_a_word: str):
    second_per_beat = 60 / bpm
    a1 = beat_in_a_word.split(" ")
    a2 = [float(x) for x in a1]  # for each element in list, turn it from str to float
    a3 = [
        format(second_per_beat * x * 4, ".4f") for x in a2
    ]  # NOTE: duration是按照四分音符是0.25这样来标的
    a4 = " ".join(a3)
    return a4


def beats_to_second(bpm, beats):
    # beats: ['0.2 | 1 | 1 0.5 | 1.5 | 0.5']
    d1 = [x.strip() for x in beats.split("|") if x.strip() != ""]  # read from '|'
    d2 = "|".join(b2s_in_a_word(bpm, x) for x in d1)  # turn list to str with '|'
    return d2


def contains_chinese(text):
    chinese_pattern = re.compile("[\u4e00-\u9fff]+")
    return chinese_pattern.search(text) is not None


def get_cmuph_for_consonan(syllable):
    consonan = syllable[0]
    if syllable == "ces" or syllable == "cem":
        return "S"
    elif syllable == "ship":
        return "SH"
    elif syllable == "yond":
        return "AA"
    elif syllable == "out":
        return "AW"
    elif syllable == "in" or syllable == "ing":
        return "IH"
    elif consonan == "c":
        return "K"
    else:
        return consonan.upper()


def get_syllable_cmuph_mapping(syllable_lst, cmuph_lst):
    if syllable_lst[0] == "enough":
        mapping = [["IH"], ["N", "AH"], ["F"]]
        return mapping
    elif syllable_lst[0] == "lovers":
        mapping = [["L", "AH"], ["V", "ER", "Z"]]
        return mapping
    elif (
        syllable_lst[0] == "for" and syllable_lst[1] == "ev" and syllable_lst[2] == "er"
    ):
        mapping = [["F", "ER"], ["EH"], ["V", "ER"]]
        return mapping
    # if syllable_lst[0] == 'ba' and syllable_lst[1] == 'by':
    #     mapping = [['B', 'EY'], ['B', 'IY']]
    #     return mapping

    # print(syllable_lst, cmuph_lst)
    elif syllable_lst[0] == "fam" and syllable_lst[1] == "i":
        # ['fam', 'i', 'ly'] ['F', 'AE', 'M', 'AH', 'L', 'IY']
        syllable_lst[0] = "fa"
        syllable_lst[1] = "mi"
    elif syllable_lst[0] == "nev" and syllable_lst[1] == "er":
        syllable_lst[0] = "ne"
        syllable_lst[1] = "ver"
    elif syllable_lst[0] == "ev" and syllable_lst[1] == "er":
        syllable_lst[0] = "e"
        syllable_lst[1] = "ver"
    elif syllable_lst[0] == "voic" and syllable_lst[1] == "es":
        syllable_lst[0] = "voi"
        syllable_lst[1] = "ces"

    # print(syllable_lst, cmuph_lst)
    mapping = []
    idx_slb = 0
    idx_ph = 0
    phs_in_this_syllable = []
    while idx_slb != len(syllable_lst) - 1:
        # next_consonan = get_cmuph_for_consonan(syllable_lst[idx_slb+1][0])
        next_consonan = get_cmuph_for_consonan(syllable_lst[idx_slb + 1])
        if cmuph_lst[idx_ph] != next_consonan:
            phs_in_this_syllable.append(cmuph_lst[idx_ph])
        else:
            mapping.append(phs_in_this_syllable)
            idx_slb += 1
            phs_in_this_syllable = []
            phs_in_this_syllable.append(cmuph_lst[idx_ph])
        idx_ph += 1
    while idx_ph != len(cmuph_lst):
        phs_in_this_syllable.append(cmuph_lst[idx_ph])
        idx_ph += 1
    mapping.append(phs_in_this_syllable)
    return mapping


def cmu_dict(dict_fn):
    dict = {}
    dict["SP"] = ["<SP>"]
    dict["AP"] = ["<AP>"]
    with open(dict_fn, "r") as f:
        lines = f.readlines()
    for line in lines:
        word = line.split()[0]
        phones = line.split()[1:]
        dict[word] = phones
    return dict


class BaseSVSInfer:
    def __init__(self, hparams, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hparams = hparams
        self.device = device
        with open(os.path.join(hparams["binary_data_dir"], "phone_set.json"), "r") as f:
            self.phone_list = json.load(f)
        self.ph_encoder = TokenTextEncoder(
            None, vocab_list=self.phone_list, replace_oov=","
        )
        # self.pinyin2phs = m4singer_pinyin2ph_func()

        dict_en_fn = (
            "/Netdata/2022/zhouhl/ml_m4singer/inference/cmu_dicts/rm-lexicon-en.txt"
        )
        self.dict_en = cmu_dict(dict_en_fn)
        dict_cn_fn = (
            "/Netdata/2022/zhouhl/ml_m4singer/inference/cmu_dicts/rm-lexicon-cn.txt"
        )
        self.dict_cn = cmu_dict(dict_cn_fn)

        with open(os.path.join(hparams["binary_data_dir"], "spk_map.json"), "r") as f:
            self.spk_map = json.load(f)

        self.model = self.build_model()
        self.model.eval()
        self.model.to(self.device)
        self.vocoder = self.build_vocoder()
        self.vocoder.eval()
        self.vocoder.to(self.device)

    def build_model(self):
        raise NotImplementedError

    def forward_model(self, inp):
        raise NotImplementedError

    def build_vocoder(self):
        base_dir = hparams["vocoder_ckpt"]
        config_path = f"{base_dir}/config.yaml"
        ckpt = sorted(
            glob.glob(f"{base_dir}/model_ckpt_steps_*.ckpt"),
            key=lambda x: int(
                re.findall(f"{base_dir}/model_ckpt_steps_(\d+).ckpt", x)[0]
            ),
        )[-1]
        print("| load HifiGAN: ", ckpt)
        ckpt_dict = torch.load(ckpt, map_location="cpu")
        config = set_hparams(config_path, global_hparams=False)
        state = ckpt_dict["state_dict"]["model_gen"]
        vocoder = HifiGanGenerator(config)
        vocoder.load_state_dict(state, strict=True)
        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(self.device)
        return vocoder

    def run_vocoder(self, c, **kwargs):
        c = c.transpose(2, 1)  # [B, 80, T]
        f0 = kwargs.get("f0")  # [B, T]
        if f0 is not None and hparams.get("use_nsf"):
            # f0 = torch.FloatTensor(f0).to(self.device)
            y = self.vocoder(c, f0).view(-1)
        else:
            y = self.vocoder(c).view(-1)
            # [T]
        return y[None]

    def preprocess_word_level_input(self, inp):
        # lyric

        # Pypinyin can't solve polyphonic words
        text_raw = inp["text"]
        # tokens = re.split(u'([\u4e00-\u9fff]+|[a-zA-Z]+)', text_raw)    # split the input string into Chinese character tokens and English word tokens withot space
        # tokens = [token for token in tokens if token.strip() != ''] # remove empty tokens
        tokens = (
            text_raw.split()
        )  # ['AP', '我喜欢你', "it's", 'the', 'circle', 'of', 'life', 'AP']
        pinyin_eng_lst = []
        language = []
        for token in tokens:
            if contains_chinese(token):
                pinyins = lazy_pinyin(token, strict=False)
                for i in pinyins:
                    pinyin_eng_lst.append(i)
                    language.append(CHINESE)
            else:
                pinyin_eng_lst.append(token)
                language.append(ENGLISH)

        # Note
        note_per_word_lst = [
            x.strip() for x in inp["notes"].split("|") if x.strip() != ""
        ]
        mididur_per_word_lst = [
            x.strip() for x in inp["notes_duration"].split("|") if x.strip() != ""
        ]

        if len(pinyin_eng_lst) == len(note_per_word_lst) == len(mididur_per_word_lst):
            print("Pass word-notes check.")
        else:
            print(
                "The number of words does't match the number of notes' windows. ",
                "You should split the note(s) for each word by | mark.",
            )
            print(pinyin_eng_lst, note_per_word_lst, mididur_per_word_lst)
            print(
                len(pinyin_eng_lst), len(note_per_word_lst), len(mididur_per_word_lst)
            )
            return None
        # for x in zip(pinyin_eng_lst, note_per_word_lst, mididur_per_word_lst):
        #     print(x)

        ph_lst = []
        note_lst = []
        midi_dur_lst = []
        is_slur = []
        lang = []

        for idx, pinyin_eng in enumerate(pinyin_eng_lst):
            this_word = pinyin_eng
            lang_in_this_word = language[idx]
            note_in_this_word = note_per_word_lst[idx].split()
            midi_dur_in_this_word = mididur_per_word_lst[idx].split()
            if this_word in ["AP", "SP"]:
                # 0.0 'AP', 'SP'
                ph_lst.append(self.dict_en[this_word][0])  # converted to <AP> <SP>
                lang.append(CHINESE)
                note_lst.append(note_in_this_word[0])
                midi_dur_lst.append(midi_dur_in_this_word[0])
                is_slur.append(0)
            elif lang_in_this_word == CHINESE:
                # 1.0 中文
                phs_in_this_word = self.dict_cn[this_word]  # [['W', 'AO']
                for ph in phs_in_this_word:
                    ph_lst.append(ph)
                    lang.append(CHINESE)
                    note_lst.append(note_in_this_word[0])
                    midi_dur_lst.append(midi_dur_in_this_word[0])
                    is_slur.append(0)
                if len(note_in_this_word) > 1:
                    # print(f'========more note for {inp["id"]}========')
                    # print(f'***********{phs_in_this_word}\n{note_in_this_word}***********')
                    # ['M', 'IY', 'AW'] ['D4', 'C4']

                    # ATTENTION：有几个声母对应cmu音素中多个phone，而不是一个phone
                    # c: T S;
                    # 但是先不考虑这么细致
                    # 从第二个note开始，重复每一个对应韵母开始的cmu音素，并标记slur
                    for note, midi_dur in zip(
                        note_in_this_word[1:], midi_dur_in_this_word[1:]
                    ):
                        for ph_yunmu in phs_in_this_word[1:]:
                            ph_lst.append(ph_yunmu)
                            lang.append(CHINESE)
                            note_lst.append(note)
                            midi_dur_lst.append(midi_dur)
                            is_slur.append(1)
                    # print(f'&&&&&&&{ph_lst}\n{note_lst}\n{is_slur}&&&&&&&')
                    # for idx in range(1, len(note_in_this_word)):
                    #     ph_lst.append(phs_in_this_word[-1])
                    #     note_lst.append(note_in_this_word[idx])
                    #     midi_dur_lst.append(midi_dur_in_this_word[idx])
                    #     is_slur.append(1)
                    #     lang.append(CHINESE)
                pass
            else:
                # 2.0 英文
                doc = nlp(this_word)
                data = [
                    (token.text, token._.syllables, token._.syllables_count)
                    for token in doc
                ]
                syllable_lst = doc[
                    0
                ]._.syllables  # [('superstar', ['su', 'per', 'star'], 3)]
                cmuph_lst = self.dict_en[this_word.lower()]
                slb_ph_mapping = get_syllable_cmuph_mapping(
                    syllable_lst, cmuph_lst
                )  # [['S', 'UW'], ['P', 'ER'], ['S', 'T', 'AA', 'R']]
                # assert len(slb_ph_mapping)==doc[0]._.syllables_count, f'{slb_ph_mapping}\t{doc[0]._.syllables_count}'
                if len(slb_ph_mapping) == len(note_in_this_word) - 1:
                    new_mapping = slb_ph_mapping[:-1]
                    phs_last_slb = slb_ph_mapping[-1]
                    slb1 = phs_last_slb[:2]
                    slb2 = phs_last_slb[1:]
                    new_mapping.append(slb1)
                    new_mapping.append(slb2)
                    slb_ph_mapping = new_mapping

                if len(slb_ph_mapping) == len(note_in_this_word):
                    # 1.0 每个音节都对应一个note，则依次对应
                    for phs_in_this_slb, note, midi_dur in zip(
                        slb_ph_mapping, note_in_this_word, midi_dur_in_this_word
                    ):
                        for ph in phs_in_this_slb:
                            ph_lst.append(ph)
                            lang.append(ENGLISH)
                            note_lst.append(note)
                            midi_dur_lst.append(midi_dur)
                            is_slur.append(0)
                elif len(slb_ph_mapping) == 1 and len(slb_ph_mapping[0]) == 1:
                    # 2.0 一个单音素音节，对应多个note，则重复该单单音素音节，并标注滑音， 如'oooh', 'UW'
                    ph = slb_ph_mapping[0][0]
                    for idx, note in enumerate(note_in_this_word):
                        ph_lst.append(ph)
                        lang.append(ENGLISH)
                        note_lst.append(note)
                        midi_dur_lst.append(midi_dur_in_this_word[idx])
                        if idx == 0:
                            is_slur.append(0)
                        else:
                            is_slur.append(1)
                elif len(note_in_this_word) == 1:
                    # 3.0 一个note，多个音节（无论几个音素）
                    note = note_in_this_word[0]
                    midi_dur = midi_dur_in_this_word[0]
                    for phs_in_this_slb in slb_ph_mapping:
                        for ph in phs_in_this_slb:
                            ph_lst.append(ph)
                            lang.append(ENGLISH)
                            note_lst.append(note)
                            midi_dur_lst.append(midi_dur)
                            is_slur.append(0)

                # assert len(slb_ph_mapping)==len(note_in_this_word), f'{slb_ph_mapping}\t{note_in_this_word}'

                # # NOTE: 每个英文单词后面都加一个sp
                # ph_lst.append('<SP>')
                # lang.append(ENGLISH)
                # note_lst.append('rest')
                # midi_dur_lst.append(0)
                # is_slur.append(0)

        # for x in inp.items():
        #     print(x)
        # for x in zip(ph_lst, lang, note_lst, midi_dur_lst, is_slur):
        #     print(x)

        # print(f'========={ph_lst}=======')
        # #### note: missing phonemes: TH Y IH DH V
        # def replace_en_with_cn(ph_lst):
        #     new_ph_lst = []
        #     for i in ph_lst:
        #         if i == 'TH':
        #             new_ph_lst.append('S')
        #         elif i == 'Y':
        #             new_ph_lst.append('IY')
        #         elif i == 'IH':
        #             new_ph_lst.append('AY')
        #         elif i == 'DH':
        #             new_ph_lst.append('Z')
        #         elif i == 'V':
        #             new_ph_lst.append('W')
        #         else:
        #             new_ph_lst.append(i)
        #     return new_ph_lst
        # ph_lst = replace_en_with_cn(ph_lst)
        # print(f'========={ph_lst}=======')
        ph_seq = " ".join(ph_lst)
        if len(ph_lst) == len(note_lst) == len(midi_dur_lst):
            print("Pass word-notes check.")
        else:
            print(
                "The number of words does't match the number of notes' windows. ",
                "You should split the note(s) for each word by | mark.",
            )
            return None
        speechsing = 1
        return ph_seq, note_lst, midi_dur_lst, is_slur, lang, speechsing

    def preprocess_phoneme_level_input(self, inp):
        #     ph_seq = inp['ph_seq']
        #     note_lst = inp['note_seq'].split()
        #     midi_dur_lst = inp['note_dur_seq'].split()
        #     is_slur = [float(x) for x in inp['is_slur_seq'].split()]
        #     lang = [float(x) for x in inp['lang_seq'].split()]
        #     print(len(note_lst), len(ph_seq.split()), len(midi_dur_lst))
        #     if len(note_lst) == len(ph_seq.split()) == len(midi_dur_lst):
        #         print('Pass word-notes check.')
        #     else:
        #         print('The number of words does\'t match the number of notes\' windows. ',
        #               'You should split the note(s) for each word by | mark.')
        #         return None
        #     return ph_seq, note_lst, midi_dur_lst, is_slur, lang
        pass

    def preprocess_input(self, inp, input_type="word"):
        item_name = inp.get("item_name", "<ITEM_NAME>")
        spk_name = inp.get("spk_name", "Tenor-1")

        # single spk
        spk_id = self.spk_map[spk_name]

        # get ph seq, note lst, midi dur lst, is slur lst.
        if input_type == "word":
            ret = self.preprocess_word_level_input(inp)
        elif input_type == "phoneme":
            ret = self.preprocess_phoneme_level_input(inp)
        else:
            print("Invalid input type.")
            return None

        if ret:
            ph_seq, note_lst, midi_dur_lst, is_slur, lang, speechsing = ret
            with open(preprocess_ret_fn, "a") as f:
                f.write(f"{ph_seq}|{note_lst}|{midi_dur_lst}|{is_slur}|{lang}\n")
        else:
            print("==========> Preprocess_word_level or phone_level input wrong.")
            return None

        # convert note lst to midi id; convert note dur lst to midi duration
        try:
            midis = [
                librosa.note_to_midi(x.split("/")[0]) if x != "rest" else 0
                for x in note_lst
            ]
            midi_dur_lst = [float(x) for x in midi_dur_lst]
        except Exception as e:
            print(e)
            print("Invalid Input Type.")
            return None

        try:
            ph_token = self.ph_encoder.encode(ph_seq)
        except:
            print(f"PH_SEQ IS {ph_seq}")
            print(f"PH_LIST FOR ENCODER IS {self.phone_list}")
            for i in ph_seq.split():
                if i not in self.phone_list:
                    print(f"{i} is not in training phone set")

        item = {
            "item_name": item_name,
            "text": inp["text"],
            "ph": ph_seq,
            "spk_id": spk_id,
            "ph_token": ph_token,
            "pitch_midi": np.asarray(midis),
            "midi_dur": np.asarray(midi_dur_lst),
            "is_slur": np.asarray(is_slur),
            "lang": np.asarray(lang),
            "speechsing": speechsing,
        }
        item["ph_len"] = len(item["ph_token"])
        return item

    def input_to_batch(self, item):
        item_names = [item["item_name"]]
        text = [item["text"]]
        ph = [item["ph"]]
        txt_tokens = torch.LongTensor(item["ph_token"])[None, :].to(self.device)
        txt_lengths = torch.LongTensor([txt_tokens.shape[1]]).to(self.device)
        spk_ids = torch.LongTensor([item["spk_id"]])[:].to(self.device)

        pitch_midi = torch.LongTensor(item["pitch_midi"])[
            None, : hparams["max_frames"]
        ].to(self.device)
        midi_dur = torch.FloatTensor(item["midi_dur"])[
            None, : hparams["max_frames"]
        ].to(self.device)
        is_slur = torch.LongTensor(item["is_slur"])[None, : hparams["max_frames"]].to(
            self.device
        )
        lang = torch.LongTensor(item["lang"])[None, : hparams["max_frames"]].to(
            self.device
        )
        speechsings = torch.LongTensor([item["speechsing"]])[:].to(self.device)

        batch = {
            "item_name": item_names,
            "text": text,
            "ph": ph,
            "txt_tokens": txt_tokens,
            "txt_lengths": txt_lengths,
            "spk_ids": spk_ids,
            "pitch_midi": pitch_midi,
            "midi_dur": midi_dur,
            "is_slur": is_slur,
            "lang": lang,
            "speechsing": speechsings,
        }
        return batch

    def postprocess_output(self, output):
        return output

    def infer_once(self, inp):
        inp = self.preprocess_input(
            inp, input_type=inp["input_type"] if inp.get("input_type") else "word"
        )
        output = self.forward_model(inp)
        output = self.postprocess_output(output)
        return output

    @classmethod
    def example_run(cls, inp):
        from utils.audio import save_wav

        set_hparams(print_hparams=False)
        # print(f'==**==hparams: \n {hparams} \n==**=========')
        infer_ins = cls(hparams)
        out = infer_ins.infer_once(inp)
        os.makedirs("infer_out", exist_ok=True)
        f_name = inp["spk_name"] + " | " + inp["text"]
        save_wav(out, f"infer_out/{f_name}.wav", hparams["audio_sample_rate"])

    @classmethod
    def infer_from_json(cls, inp_fn, save_path, bpm):
        set_hparams(print_hparams=True)
        infer_ins = cls(hparams)

        # with open(inp_fn, 'r', encoding='utf-8') as f:
        #     inps = [eval(line) for line in f.readlines()]
        # os.makedirs('infer_out', exist_ok=True)
        inps = json.load(open(inp_fn, "r"))  # 'word' formant

        os.makedirs(save_path, exist_ok=True)
        for inp in inps:
            if inp.get("bpm"):
                inp["notes_duration"] = beats_to_second(
                    inp["bpm"], inp["notes_duration"]
                )
            else:
                # inp['notes_duration'] = beats_to_second(bpm, inp['notes_duration'])
                pass
            # inp['notes_duration'] = beats_to_second(bpm, inp['notes_duration'])
            inp["spk_name"] = f'{inp_fn.split("/")[-2]}-1'
            out = infer_ins.infer_once(inp)
            f_name = str(inp["id"]) + "|" + inp["spk_name"] + "|" + inp["text"]
            save_wav(out, f"{save_path}/{f_name}.wav", hparams["audio_sample_rate"])


class DiffSingerE2EInfer(BaseSVSInfer):
    def build_model(self):
        model = GaussianDiffusion(
            phone_encoder=self.ph_encoder,
            out_dims=hparams["audio_num_mel_bins"],
            denoise_fn=DIFF_DECODERS[hparams["diff_decoder_type"]](hparams),
            timesteps=hparams["timesteps"],
            K_step=hparams["K_step"],
            loss_type=hparams["diff_loss_type"],
            spec_min=hparams["spec_min"],
            spec_max=hparams["spec_max"],
        )
        model.eval()
        load_ckpt(model, hparams["work_dir"], "model")

        if hparams.get("pe_enable") is not None and hparams["pe_enable"]:
            self.pe = PitchExtractor().to(self.device)
            utils.load_ckpt(self.pe, hparams["pe_ckpt"], "model", strict=True)
            self.pe.eval()
        return model

    def forward_model(self, inp):
        sample = self.input_to_batch(inp)
        txt_tokens = sample["txt_tokens"]  # [B, T_t]
        spk_id = sample.get("spk_ids")
        with torch.no_grad():
            # import joblib
            # joblib.dump(dict(txt_tokens=txt_tokens, spk_embed=spk_id, ref_mels=None, infer=True,
            #                     pitch_midi=sample['pitch_midi'], midi_dur=sample['midi_dur'],
            #                     is_slur=sample['is_slur']), 'infer_out/diff_wrong')
            output = self.model(
                txt_tokens,
                spk_embed=spk_id,
                ref_mels=None,
                infer=True,
                pitch_midi=sample["pitch_midi"],
                midi_dur=sample["midi_dur"],
                is_slur=sample["is_slur"],
                lang=sample["lang"],
                speechsing=sample["speechsing"],
            )
            # import joblib
            # output = self.model(**joblib.load('infer_out/shiyao.pt'))
            mel_out = output["mel_out"]  # [B, T,80]
            if hparams.get("pe_enable") is not None and hparams["pe_enable"]:
                f0_pred = self.pe(mel_out)["f0_denorm_pred"]  # pe predict from Pred mel
            else:
                f0_pred = output["f0_denorm"]
            wav_out = self.run_vocoder(mel_out, f0=f0_pred)
        wav_out = wav_out.cpu().numpy()
        return wav_out[0]


if __name__ == "__main__":
    # kinds = ['cn', 'en', 'mix']
    kinds = ["woheni"]
    # 不同的声部使用不同的note作为乐谱
    # parts = ['Bass', 'Tenor', 'Alto', 'Soprano']
    parts = ["Tenor"]
    for kind in kinds:
        for part in parts:
            inp_fn = f"/Netdata/2022/zhouhl/ml_m4singer3.0/data/bisinger-testcase/ver6/{part}/{kind}.json"
            save_path = f"/Netdata/2022/zhouhl/bisinger_mos/system-3/{kind}"
            preprocess_ret_fn = f"/Netdata/2022/zhouhl/bisinger_mos/system-3/{kind}.txt"
            if os.path.exists(preprocess_ret_fn):
                os.remove(preprocess_ret_fn)
            # bpm=110 # not used here
            DiffSingerE2EInfer.infer_from_json(inp_fn, save_path, bpm=None)
