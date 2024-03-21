import textgrid
import parselmouth
import numpy as np
import json
import os
from functools import reduce
from tqdm import tqdm

dict_item2txt = {}
with open("data/mfa_preparation/output_cn.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        id, text, pinyin = line.split("|")
        dict_item2txt[id] = text.replace(" ", "")


def get_cn_info_from_tg_wav(tg_path, wav_path, json_fn):
    id, ext = os.path.splitext(tg_path.split("/")[-1])
    tg = textgrid.TextGrid.fromFile(tg_path)
    snd = parselmouth.Sound(wav_path)
    # 1. 得到word和phone的映射关系
    # 2. 得到words、notes、note_durations
    # 3. 得到phones和phone_duraions
    # 4. 根据映射关系，对notes和note_durations进行相应的复制
    word_tier = tg[0]
    phone_tier = tg[1]
    word_phone_mapping_list = []  # 第i个元素的值意味着第i个word对应的phone的index
    words = []
    word_boundary = []
    notes = []
    note_durations = []  # not duplicated

    for i, interval in enumerate(word_tier):
        words.append("<SP>" if interval.mark == "<eps>" else interval.mark)
        word_start_time = interval.minTime
        word_end_time = interval.maxTime
        note_durations.append(round(word_end_time - word_start_time, 4))
        if interval.mark == "<eps>":
            notes.append(0)
        else:
            # Extract the corresponding signal for the current interval
            word_signal = snd.extract_part(
                from_time=word_start_time, to_time=word_end_time
            )
            # Compute the f0 using Parselmouth
            pitch = word_signal.to_pitch_cc()
            frequencies = np.array(pitch.selected_array["frequency"])
            # print(f'frequencies: {frequencies}')
            nonzero_frequencies = frequencies[frequencies > 0]  # 获取所有非零频率值
            mean_nonzero_f0 = np.mean(nonzero_frequencies)  # 计算平均非零f0
            mean_nonzero_f0_no_nan = np.nan_to_num(mean_nonzero_f0)  # 将NaN值替换为0
            # 将f0转换为midi音高
            notes.append(
                round(69 + 12 * np.log2(mean_nonzero_f0_no_nan / 440))
                if mean_nonzero_f0_no_nan != 0
                else 0
            )
            # print(f'f0 for {interval.mark}: {mean_nonzero_f0_no_nan:.2f} Hz') # Print the f0 value

        t = []
        phones = []
        phone_duraions = []
        for j, phone in enumerate(phone_tier):
            phones.append("<SP>" if phone.mark == "sil" else phone.mark)
            phone_start_time = phone.minTime
            phone_end_time = phone.maxTime
            phone_duraions.append(round(phone_end_time - phone_start_time, 4))
            if phone_start_time >= word_start_time and phone_end_time <= word_end_time:
                t.append(j)
        word_phone_mapping_list.append(t)

    # print(word_phone_mapping_list)
    # # [[0], [1, 2, 3], [4, 5, 6], [7, 8], [9, 10], [11, 12, 13], [14, 15, 16], [17], [18, 19], [20, 21], [22, 23], [24, 25, 26], [27, 28, 29], [30, 31], [32, 33, 34], [35]]
    # print(phones)
    # # ['sil', 'X', 'IY', 'N', 'D', 'UH', 'NG', 'B', 'UW', 'R', 'UW', 'X', 'IY', 'NG', 'D', 'UH', 'NG', 'sil', 'W', 'AO', 'B', 'UW', 'T', 'AY', 'SH', 'AE', 'N', 'CH', 'AE', 'NG', 'M', 'AY', 'M', 'AH', 'NG', 'sil']
    # print(phone_duraions)
    # # [0.28, 0.13, 0.04, 0.11, 0.05, 0.14, 0.11, 0.08, 0.05, 0.05, 0.09, 0.15, 0.08, 0.12, 0.06, 0.12, 0.1, 0.37, 0.05, 0.05, 0.09, 0.04, 0.12, 0.08, 0.09, 0.06, 0.04, 0.06, 0.06, 0.04, 0.06, 0.1, 0.08, 0.16, 0.09, 0.25]
    # print(notes)
    # # [0, 64, 62, 64, 62, 58, 63, 0, 59, 56, 64, 63, 58, 58, 56, 0]
    # print(words)
    # # ['<eps>', 'xin', 'dong', 'bu', 'ru', 'xing', 'dong', '<eps>', 'wo', 'bu', 'tai', 'shan', 'chang', 'mai', 'meng', '<eps>']
    # print(note_durations)
    # # [0.28, 0.28, 0.3, 0.13, 0.14, 0.35, 0.28, 0.37, 0.1, 0.13, 0.2, 0.19, 0.16, 0.16, 0.33, 0.25]

    # 计算word_boundary
    for word_unit in word_phone_mapping_list:
        word_boundary.append([0] * (len(word_unit) - 1) + [1])
    word_boundary[0] = [0]

    reassure_note_dur = []
    for ph_idxs, ph_dur in zip(word_phone_mapping_list, phone_duraions):
        reassure_note_dur.append(round(sum(phone_duraions[i] for i in ph_idxs), 4))
    # print(reassure_note_dur)
    # # [0.28, 0.28, 0.3, 0.13, 0.14, 0.35, 0.28, 0.37, 0.1, 0.13, 0.2, 0.19, 0.16, 0.16, 0.33, 0.25]
    assert reassure_note_dur == note_durations, "wrong note durations"

    notes_repeated = []
    note_durations_repeated = []
    for i in range(len(notes)):
        notes_repeated.extend([notes[i]] * len(word_phone_mapping_list[i]))
    for i in range(len(notes)):
        note_durations_repeated.extend(
            [note_durations[i]] * len(word_phone_mapping_list[i])
        )
    # print(notes_repeated)
    # # [0, 64, 64, 64, 62, 62, 62, 64, 64, 62, 62, 58, 58, 58, 63, 63, 63, 0, 59, 59, 56, 56, 64, 64, 63, 63, 63, 58, 58, 58, 58, 58, 56, 56, 56, 0]
    # print(note_durations_repeated)
    # # [0.28, 0.28, 0.28, 0.28, 0.3, 0.3, 0.3, 0.13, 0.13, 0.14, 0.14, 0.35, 0.35, 0.35, 0.28, 0.28, 0.28, 0.37, 0.1, 0.1, 0.13, 0.13, 0.2, 0.2, 0.19, 0.19, 0.19, 0.16, 0.16, 0.16, 0.16, 0.16, 0.33, 0.33, 0.33, 0.25]

    # 将句子开头和结尾处的sp都换为ap
    words[0] = "<AP>" if words[0] == "<SP>" else words[0]
    words[-1] = "<AP>" if words[-1] == "<SP>" else words[-1]
    phones[0] = "<AP>" if phones[0] == "<SP>" else phones[0]
    phones[-1] = "<AP>" if phones[-1] == "<SP>" else phones[-1]

    assert (
        len(phones)
        == len(phone_duraions)
        == len(notes_repeated)
        == len(note_durations_repeated)
    ), "mismatch length between phones and notes"
    info = {
        "lang": 1,
        "item_name": f"db4#cn#{id}",
        "txt": dict_item2txt[id],
        "words": " ".join(words).strip(),
        "phs": phones,
        "is_slur": [0] * len(phones),
        "ph_dur": phone_duraions,
        "notes": notes_repeated,
        "notes_dur": note_durations_repeated,
        "word_boundary": reduce(lambda x, y: x + y, word_boundary),
    }
    # print(info)
    with open(json_fn, "a", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False)
        f.write("\n")
    return info


if __name__ == "__main__":
    root_tg = "data/mfa_workspace/db4-cn/output"
    root_wav = "data/mfa_workspace/db4-cn/input"

    # 如果文件存在，则删除文件
    json_fn = "data/meta/db4cn-wdb.json"
    if os.path.exists(json_fn):
        os.remove(json_fn)
    # cnt = 0
    for root, dirs, files in os.walk(root_tg):
        for file in tqdm(sorted(files)):
            # cnt += 1
            # if cnt == 10:
            #     break
            tg_path = os.path.join(root_tg, file)
            id, ext = os.path.splitext(file)
            wav_fn = id + ".wav"  # 构造对应的.wav文件名
            wav_path = os.path.join(root_wav, wav_fn)

            info = get_cn_info_from_tg_wav(tg_path, wav_path, json_fn)
            # print(info)
