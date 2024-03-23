import glob
import os
import re

import librosa
import numpy as np
import torch
from inference.m4singer.m4singer.map import m4singer_pinyin2ph_func
from modules.hifigan.hifigan import HifiGanGenerator
from pypinyin import Style, lazy_pinyin, pinyin
from vocoders.hifigan import HifiGAN

from utils import load_ckpt
from utils.hparams import hparams, set_hparams
from utils.text_encoder import TokenTextEncoder


class BaseSVSInfer:
    def __init__(self, hparams, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hparams = hparams
        self.device = device

        phone_list = [
            "<AP>",
            "<SP>",
            "a",
            "ai",
            "an",
            "ang",
            "ao",
            "b",
            "c",
            "ch",
            "d",
            "e",
            "ei",
            "en",
            "eng",
            "er",
            "f",
            "g",
            "h",
            "i",
            "ia",
            "ian",
            "iang",
            "iao",
            "ie",
            "in",
            "ing",
            "iong",
            "iou",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "ong",
            "ou",
            "p",
            "q",
            "r",
            "s",
            "sh",
            "t",
            "u",
            "ua",
            "uai",
            "uan",
            "uang",
            "uei",
            "uen",
            "uo",
            "v",
            "van",
            "ve",
            "vn",
            "x",
            "z",
            "zh",
        ]
        self.ph_encoder = TokenTextEncoder(None, vocab_list=phone_list, replace_oov=",")
        self.pinyin2phs = m4singer_pinyin2ph_func()
        self.spk_map = {
            "Alto-1": 0,
            "Alto-2": 1,
            "Alto-3": 2,
            "Alto-4": 3,
            "Alto-5": 4,
            "Alto-6": 5,
            "Alto-7": 6,
            "Bass-1": 7,
            "Bass-2": 8,
            "Bass-3": 9,
            "Soprano-1": 10,
            "Soprano-2": 11,
            "Soprano-3": 12,
            "Tenor-1": 13,
            "Tenor-2": 14,
            "Tenor-3": 15,
            "Tenor-4": 16,
            "Tenor-5": 17,
            "Tenor-6": 18,
            "Tenor-7": 19,
        }

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
        # Pypinyin can't solve polyphonic words
        text_raw = inp["text"]

        # lyric
        pinyins = lazy_pinyin(text_raw, strict=False)
        ph_per_word_lst = [
            self.pinyin2phs[pinyin.strip()]
            for pinyin in pinyins
            if pinyin.strip() in self.pinyin2phs
        ]

        # Note
        note_per_word_lst = [
            x.strip() for x in inp["notes"].split("|") if x.strip() != ""
        ]
        mididur_per_word_lst = [
            x.strip() for x in inp["notes_duration"].split("|") if x.strip() != ""
        ]

        if len(note_per_word_lst) == len(ph_per_word_lst) == len(mididur_per_word_lst):
            print("Pass word-notes check.")
        else:
            print(
                "The number of words does't match the number of notes' windows. ",
                "You should split the note(s) for each word by | mark.",
            )
            print(ph_per_word_lst, note_per_word_lst, mididur_per_word_lst)
            print(
                len(ph_per_word_lst), len(note_per_word_lst), len(mididur_per_word_lst)
            )
            return None

        note_lst = []
        ph_lst = []
        midi_dur_lst = []
        is_slur = []
        for idx, ph_per_word in enumerate(ph_per_word_lst):
            # for phs in one word:
            # single ph like ['ai']  or multiple phs like ['n', 'i']
            ph_in_this_word = ph_per_word.split()

            # for notes in one word:
            # single note like ['D4'] or multiple notes like ['D4', 'E4'] which means a 'slur' here.
            note_in_this_word = note_per_word_lst[idx].split()
            midi_dur_in_this_word = mididur_per_word_lst[idx].split()
            # process for the model input
            # Step 1.
            #  Deal with note of 'not slur' case or the first note of 'slur' case
            #  j        ie
            #  F#4/Gb4  F#4/Gb4
            #  0        0
            for ph in ph_in_this_word:
                ph_lst.append(ph)
                note_lst.append(note_in_this_word[0])
                midi_dur_lst.append(midi_dur_in_this_word[0])
                is_slur.append(0)
            # step 2.
            #  Deal with the 2nd, 3rd... notes of 'slur' case
            #  j        ie         ie
            #  F#4/Gb4  F#4/Gb4    C#4/Db4
            #  0        0          1
            if (
                len(note_in_this_word) > 1
            ):  # is_slur = True, we should repeat the YUNMU to match the 2nd, 3rd... notes.
                for idx in range(1, len(note_in_this_word)):
                    ph_lst.append(ph_in_this_word[-1])
                    note_lst.append(note_in_this_word[idx])
                    midi_dur_lst.append(midi_dur_in_this_word[idx])
                    is_slur.append(1)
        ph_seq = " ".join(ph_lst)

        if len(ph_lst) == len(note_lst) == len(midi_dur_lst):
            print(len(ph_lst), len(note_lst), len(midi_dur_lst))
            print("Pass word-notes check.")
        else:
            print(
                "The number of words does't match the number of notes' windows. ",
                "You should split the note(s) for each word by | mark.",
            )
            return None
        return ph_seq, note_lst, midi_dur_lst, is_slur

    def preprocess_phoneme_level_input(self, inp):
        ph_seq = inp["ph_seq"]
        note_lst = inp["note_seq"].split()
        midi_dur_lst = inp["note_dur_seq"].split()
        is_slur = [float(x) for x in inp["is_slur_seq"].split()]
        print(len(note_lst), len(ph_seq.split()), len(midi_dur_lst))
        if len(note_lst) == len(ph_seq.split()) == len(midi_dur_lst):
            print("Pass word-notes check.")
        else:
            print(
                "The number of words does't match the number of notes' windows. ",
                "You should split the note(s) for each word by | mark.",
            )
            return None
        return ph_seq, note_lst, midi_dur_lst, is_slur

    def preprocess_input(self, inp, input_type="word"):
        """

        :param inp: {'text': str, 'item_name': (str, optional), 'spk_name': (str, optional)}
        :return:
        """

        item_name = inp.get("item_name", "<ITEM_NAME>")
        spk_name = inp.get("spk_name", "Alto-1")

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
            ph_seq, note_lst, midi_dur_lst, is_slur = ret
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

        ph_token = self.ph_encoder.encode(ph_seq)
        item = {
            "item_name": item_name,
            "text": inp["text"],
            "ph": ph_seq,
            "spk_id": spk_id,
            "ph_token": ph_token,
            "pitch_midi": np.asarray(midis),
            "midi_dur": np.asarray(midi_dur_lst),
            "is_slur": np.asarray(is_slur),
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
        infer_ins = cls(hparams)
        out = infer_ins.infer_once(inp)
        os.makedirs("infer_out", exist_ok=True)
        f_name = inp["spk_name"] + " | " + inp["text"]
        save_wav(out, f"infer_out/{f_name}.wav", hparams["audio_sample_rate"])
