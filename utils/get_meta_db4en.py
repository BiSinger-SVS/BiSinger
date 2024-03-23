import json
import os

import numpy as np
import parselmouth
import textgrid

dict_item2txt = {}
with open("data/mfa_preparation/output_en.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        id, en_text, cmu_phs = line.split("|")
        dict_item2txt[id] = en_text


def get_en_info_from_tg_wav(tg_path, wav_path, json_fn):
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
    notes = []
    note_durations = []  # not duplicated

    for i, interval in enumerate(word_tier):
        words.append("<SP>" if interval.mark == "" else interval.mark)
        word_start_time = interval.minTime
        word_end_time = interval.maxTime
        note_durations.append(round(word_end_time - word_start_time, 4))
        if interval.mark == "":
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
            phones.append("<SP>" if phone.mark == "" else phone.mark)
            phone_start_time = phone.minTime
            phone_end_time = phone.maxTime
            phone_duraions.append(round(phone_end_time - phone_start_time, 4))
            if phone_start_time >= word_start_time and phone_end_time <= word_end_time:
                t.append(j)
        word_phone_mapping_list.append(t)

    # print(word_phone_mapping_list)
    # # [[0], [1, 2, 3], [4, 5, 6], [7], [8, 9], [10], [11, 12, 13, 14], [15], [16, 17, 18], [19], [20, 21, 22, 23], [24]]
    # print(phones)
    # # ['<SP>', 'S', 'T', 'EY', 'K', 'UW', 'L', '<SP>', 'HH', 'IY', '<SP>', 'AE', 'D', 'IH', 'D', '<SP>', 'W', 'IH', 'DH', 'AH', 'S', 'M', 'AY', 'L', '<SP>']
    # print(phone_duraions)
    # # [0.27, 0.12, 0.06, 0.14, 0.15, 0.1, 0.31, 0.03, 0.07, 0.08, 0.06, 0.15, 0.05, 0.14, 0.06, 0.25, 0.06, 0.03, 0.06, 0.07, 0.14, 0.07, 0.21, 0.17, 0.2904]
    # print(notes)
    # # [0, 63, 62, 0, 58, 0, 58, 0, 58, 57, 57, 0]
    # print(words)
    # # ['<SP>', 'stay', 'cool', '<SP>', 'he', '<SP>', 'added', '<SP>', 'with', 'a', 'smile', '<SP>']
    # print(note_durations)
    # # [0.27, 0.32, 0.56, 0.03, 0.15, 0.06, 0.4, 0.25, 0.15, 0.07, 0.59, 0.2904]

    reassure_note_dur = []
    for ph_idxs, ph_dur in zip(word_phone_mapping_list, phone_duraions):
        reassure_note_dur.append(round(sum(phone_duraions[i] for i in ph_idxs), 4))
    # print(reassure_note_dur)
    # # [0.27, 0.32, 0.56, 0.03, 0.15, 0.06, 0.4, 0.25, 0.15, 0.07, 0.59, 0.2904]
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
    # # [0, 63, 63, 63, 62, 62, 62, 0, 58, 58, 0, 58, 58, 58, 58, 0, 58, 58, 58, 57, 57, 57, 57, 57, 0]
    # print(note_durations_repeated)
    # # [0.27, 0.32, 0.32, 0.32, 0.56, 0.56, 0.56, 0.03, 0.15, 0.15, 0.06, 0.4, 0.4, 0.4, 0.4, 0.25, 0.15, 0.15, 0.15, 0.07, 0.59, 0.59, 0.59, 0.59, 0.2904]

    assert (
        len(phones)
        == len(phone_duraions)
        == len(notes_repeated)
        == len(note_durations_repeated)
    ), "mismatch length between phones and notes"
    info = {
        "lang": 0,
        "item_name": f"db4#en#{id}",
        "txt": dict_item2txt[id],
        "words": " ".join(words).strip(),
        "phs": phones,
        "is_slur": [0] * len(phones),
        "ph_dur": phone_duraions,
        "notes": notes_repeated,
        "notes_dur": note_durations_repeated,
    }
    # print(info)
    with open(json_fn, "a", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False)
        f.write("\n")
    return info


if __name__ == "__main__":
    root_tg = "data/mfa_workspace/db4-en/output"
    root_wav = "data/mfa_workspace/db4-en/input"

    # 如果文件存在，则删除文件
    json_fn = "data/meta/db4en.json"
    if os.path.exists(json_fn):
        os.remove(json_fn)
    # cnt = 0
    for root, dirs, files in os.walk(root_tg):
        for file in sorted(files):
            # cnt += 1
            # if cnt == 10:
            #     break
            tg_path = os.path.join(root_tg, file)
            id, ext = os.path.splitext(file)
            wav_fn = id + ".wav"  # 构造对应的.wav文件名
            wav_path = os.path.join(root_wav, wav_fn)

            try:
                info = get_en_info_from_tg_wav(tg_path, wav_path, json_fn)
                print(info)
            except:
                print(f"WARNING INFO: there are some mistakes when deal with {wav_fn}")
