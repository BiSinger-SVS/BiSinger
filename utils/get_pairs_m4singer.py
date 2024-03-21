import json
import os
from pypinyin import lazy_pinyin

ALL_YUNMU = [
    "a",
    "ai",
    "an",
    "ang",
    "ao",
    "e",
    "ei",
    "en",
    "eng",
    "er",
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
    "o",
    "ong",
    "ou",
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
]

# 为每个说话人在mfa-m4singer数据目录下创建对应的文件夹
singers = [
    "Alto-1",
    "Alto-2",
    "Alto-3",
    "Alto-4",
    "Alto-5",
    "Alto-6",
    "Alto-7",
    "Bass-1",
    "Bass-2",
    "Bass-3",
    "Soprano-1",
    "Soprano-2",
    "Soprano-3",
    "Tenor-1",
    "Tenor-2",
    "Tenor-3",
    "Tenor-4",
    "Tenor-5",
    "Tenor-6",
    "Tenor-7",
]
m4_mfa_path = "data/mfa_workspace/m4singer/input"
for singer in singers:
    dir_path = os.path.join(m4_mfa_path, singer)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        pass


raw_data_dir = "/Netdata/AudioData/m4singer"
song_items = json.load(open(os.path.join(raw_data_dir, "meta.json")))  # [list of dict]

# 如果文件存在，则删除文件
pairs_fn = "m4singer_pairs.txt"
if os.path.exists(pairs_fn):
    os.remove(pairs_fn)
with open(pairs_fn, "w") as f:
    # cnt = 0
    for song_item in song_items:
        # cnt += 1
        # if cnt == 10:
        #     break
        # read from meta.json
        item_name = raw_item_name = song_item["item_name"]
        singer, song_name, sent_id = item_name.split("#")
        item2wavfn = f"{raw_data_dir}/{singer}#{song_name}/{sent_id}.wav"
        item2txt = song_item["txt"]
        item2ph = " ".join(song_item["phs"])
        item2ph_durs = song_item["ph_dur"]
        item2midi = song_item["notes"]
        item2midi_dur = song_item["notes_dur"]
        item2is_slur = song_item["is_slur"]
        item2wdb = [
            (
                1
                if (
                    0 < i < len(song_item["phs"]) - 1
                    and p in ALL_YUNMU + ["<SP>", "<AP>"]
                )
                or i == len(song_item["phs"]) - 1
                else 0
            )
            for i, p in enumerate(song_item["phs"])
        ]

        # get pinyin
        item2pinyin = " ".join(lazy_pinyin(item2txt))

        # cp wav file
        wav_fn = os.path.join(m4_mfa_path, singer, f"{song_name}#{sent_id}.wav")
        os.system(f'cp "{item2wavfn}" "{wav_fn}"')

        # write transprition for wav
        txt_fn = os.path.join(m4_mfa_path, singer, f"{song_name}#{sent_id}.txt")
        os.system(f'echo "{item2pinyin}" > "{txt_fn}"')

        f.write(
            f"{singer}|{song_name}|{sent_id}|{item2txt}|{item2pinyin}|{item2wdb}|{item2is_slur}|{item2ph}|{item2ph_durs}|{item2midi}|{item2midi_dur}\n"
        )
