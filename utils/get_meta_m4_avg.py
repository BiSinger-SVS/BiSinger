import json
import os
import os.path as osp
from functools import reduce

from pypinyin import lazy_pinyin
from tqdm import tqdm

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
# original m4singer meta.json
ROOT = "/Netdata/AudioData/m4singer"
song_items = json.load(
    open(os.path.join(ROOT, "meta.json"))
)  # [list of dict], original json file

# phone set transition
m4py2cmu_fn = "assets/pinyin_cmu_map.txt"
with open(osp.join(ROOT, m4py2cmu_fn), "r") as fid:
    lines = fid.readlines()
dict_m4_py2cmu = dict(
    [
        (split[0].lower(), split[1:])
        for split in [line.strip().split() for line in lines]
    ]
)
dict_m4_py2cmu["<AP>"] = ["<AP>"]
dict_m4_py2cmu["<SP>"] = ["<SP>"]


# new m4singer meta.json based on CMU phone set
json_fn = "data/meta/m4-avg.json"  # averaged json file
# 如果文件存在，则删除文件
if os.path.exists(json_fn):
    os.remove(json_fn)


def slur_json_tg(is_slur):
    re = []
    cur_lst = []
    for idx, val in enumerate(is_slur):
        if val == 1:
            cur_lst.append(idx)
            continue
        if cur_lst:
            re.append(cur_lst)
            cur_lst = []
        cur_lst.append(idx)
    if cur_lst:
        re.append(cur_lst)
    return re


for item in tqdm(song_items):
    phs_t = [dict_m4_py2cmu[ph] for ph in item["phs"]]
    # 18 [['<SP>'], ['<SP>'], ['UW', 'AO'], ['T', 'S'], ['AY'], ['AY'], ['IY', 'AE', 'NG'], ['IY', 'AE', 'NG'], ['IY', 'AE', 'NG'], ['G'], ['UW', 'AE', 'NG'], ['HH'], ['UW', 'IY'], ['HH'], ['AW'], ['AW'], ['AW'], ['<SP>']]
    is_slur_t = []
    notes_t = []
    notes_dur_t = []
    ph_dur_t = []
    wdb_t = []
    for i, ph in enumerate(phs_t):
        is_slur_t.append([item["is_slur"][i]] * len(phs_t[i]))
        # 18 [[0], [0], [0, 0], [0, 0], [0], [1], [0, 0, 0], [1, 1, 1], [1, 1, 1], [0], [0, 0, 0], [0], [0, 0], [0], [0], [1], [1], [0]]
        notes_t.append([item["notes"][i]] * len(phs_t[i]))
        # 18 [[0], [0], [63, 63], [64, 64], [64], [65], [65, 65, 65], [63, 63, 63], [64, 64, 64], [63], [63, 63, 63], [61], [61, 61], [61], [61], [63], [60], [0]]
        notes_dur_t.append([item["notes_dur"][i]] * len(phs_t[i]))
        # 18 [[0.14], [0.34], [0.25, 0.25], [0.4391, 0.4391], [0.4391], [0.2109], [0.105, 0.105, 0.105], [0.4205, 0.4205, 0.4205], [0.2245, 0.2245, 0.2245], [0.23], [0.23, 0.23, 0.23], [0.4], [0.4, 0.4], [0.3545], [0.3545], [0.2296], [0.8359], [0.13]]
        ph_dur_t.append([round(item["ph_dur"][i] / len(phs_t[i]), 4)] * len(phs_t[i]))
        # 18 [[0.14], [0.34], [0.125, 0.125], [0.125, 0.125], [0.1891], [0.2109], [0.035, 0.035, 0.035], [0.1402, 0.1402, 0.1402], [0.0748, 0.0748, 0.0748], [0.11], [0.04, 0.04, 0.04], [0.17], [0.115, 0.115], [0.25], [0.1045], [0.2296], [0.8359], [0.13]]

    pinyin_wdb = [
        (
            1
            if (0 < i < len(item["phs"]) - 1 and p in ALL_YUNMU + ["<SP>", "<AP>"])
            or i == len(item["phs"]) - 1
            else 0
        )
        for i, p in enumerate(item["phs"])
    ]
    for idx, flag in enumerate(pinyin_wdb):
        if flag == 0:
            wdb_t.append([0] * len(phs_t[idx]))
        else:
            wdb_t.append([0] * (len(phs_t[idx]) - 1) + [1])

    # 根据is_slur修正word boundary
    re = slur_json_tg(
        item["is_slur"]
    )  # [[0], [1], [2], [3], [4, 5], [6], [7], [8], [9, 10], [11], [12], [13]]
    for i in re:
        if len(i) > 1:
            for idx_pinyin in i:
                wdb_t[idx_pinyin] = [0] * len(wdb_t[idx_pinyin])
            wdb_t[i[-1]][-1] = 1

    info = {
        "lang": 1,
        "item_name": item["item_name"],
        "txt": item["txt"],
        "words": " ".join(lazy_pinyin(item["txt"])),  # not read from aligned textgrid
        "phs": reduce(lambda x, y: x + y, phs_t),
        "is_slur": reduce(lambda x, y: x + y, is_slur_t),
        "ph_dur": reduce(lambda x, y: x + y, ph_dur_t),
        "notes": reduce(lambda x, y: x + y, notes_t),
        "notes_dur": reduce(lambda x, y: x + y, notes_dur_t),
        "word_boundary": reduce(lambda x, y: x + y, wdb_t),
    }
    with open(json_fn, "a", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False)
        f.write("\n")
