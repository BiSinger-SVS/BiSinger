import soundfile as sf
import numpy as np
import pyworld as pw
import math

# import matplotlib.pyplot as plt
import json
import random
import os
from tqdm import tqdm


# 根据note转为freq
def note_to_freq(note):
    # split the alphabets and numbers
    note = list(note)
    note_name = note[0]
    note_num = int(note[1])
    # get the frequency
    if note_name == "C":
        freq = 16.35 * (2**note_num)
    elif note_name == "D":
        freq = 18.35 * (2**note_num)
    elif note_name == "E":
        freq = 20.60 * (2**note_num)
    elif note_name == "F":
        freq = 21.83 * (2**note_num)
    elif note_name == "G":
        freq = 24.50 * (2**note_num)
    elif note_name == "A":
        freq = 27.50 * (2**note_num)
    elif note_name == "B":
        freq = 30.87 * (2**note_num)
    else:
        raise ValueError("Invalid note name")
    return freq


# 调用note_to_freq函数，根据用1234567表示的CDEFGAB，转为freq
def num_to_freq(num, p):
    if num == 1:
        freq = note_to_freq("C{}".format(p))
    elif num == 2:
        freq = note_to_freq("D{}".format(p))
    elif num == 3:
        freq = note_to_freq("E{}".format(p))
    elif num == 4:
        freq = note_to_freq("F{}".format(p))
    elif num == 5:
        freq = note_to_freq("G{}".format(p))
    elif num == 6:
        freq = note_to_freq("A{}".format(p))
    elif num == 7:
        freq = note_to_freq("B{}".format(p))
    else:
        raise ValueError("Invalid note number")
    return freq


# 根据world重新合成的wav的freq，重新映射为midi_number
def freq_to_midi(frequency):
    midi_number = 69 + 12 * math.log2(frequency / 440)
    return round(midi_number)


# 定义常见的几种和弦
# 15634125
canon = [523.25, 392.00, 440.00, 329.63, 349.23, 523.25, 293.66, 392.00]
# 4536251
common_1 = [num_to_freq(int(num), 4) for num in list("4536251")]
# 456
common_2 = [num_to_freq(int(num), 4) for num in list("456")]
# 17654325
common_3 = [num_to_freq(int(num), 4) for num in list("17654325")]
# 1563451
common_4 = [num_to_freq(int(num), 4) for num in list("1563451")]
# 62514273
common_5 = [num_to_freq(int(num), 4) for num in list("62514273")]
# 63451
common_6 = [num_to_freq(int(num), 4) for num in list("63451")]
# 1234567
common_7 = [num_to_freq(int(num), 4) for num in list("1234567")]
# 7654321
common_8 = [num_to_freq(int(num), 4) for num in list("7654321")]
# 6415
common_9 = [num_to_freq(int(num), 4) for num in list("6415")]

chords = [
    common_1,
    common_2,
    common_3,
    common_4,
    common_5,
    common_6,
    common_7,
    common_8,
    common_9,
    canon,
]
# print([freq_to_midi(f) for f in canon])


def process_item(item, ori_wav_path, tgt_wav_path):
    # "item_name": "db4#en#300154"
    singer, song, item_id = item["item_name"].split("#")
    chord = random.choice(chords)

    # Load interval from data
    note_dur = item["notes_dur"]
    ph_dur = item["ph_dur"]
    notes = item["notes"]
    unique_note_dur = []  # List to store unique note_dur values
    s = 0
    count = 0
    rep_count = []
    sp_ids = []
    length = len(note_dur)
    idx = 0
    for _ in range(length):
        if idx >= length:
            break
        dur = note_dur[idx]
        for k in range(length - idx):
            # print(idx, k)
            s = s + ph_dur[k + idx]
            count = count + 1
            # print(s)
            if math.isclose(s, dur, abs_tol=0.001):
                unique_note_dur.append(dur)
                rep_count.append(count)
                if notes[idx] == 0:
                    sp_ids.append(1)
                else:
                    sp_ids.append(0)
                s = 0
                count = 0
                # print(idx, k)
                idx = idx + k + 1
                break
    # based on the note_dur, get the new f0 from repeatting canon
    new_f0 = []
    delta = 0
    for j in range(len(unique_note_dur)):
        if sp_ids[j] == 1:
            new_f0.extend([0])
            delta += 1
            continue
        idx = (j - delta) % len(chord)
        new_f0.extend([chord[idx]])
    # print(rep_count)
    # shift the f0 according to the note_dur and new_f0
    # _, time_axis = pw.dio(x, fs)
    # f0 = pw.stonemask(x, pw.dio(x, fs)[0], pw.dio(x, fs)[1], fs)

    # Load the audio signal
    x, fs = sf.read(f"{ori_wav_path}/{item_id}.wav")
    x = np.ascontiguousarray(x)
    f0, sp, ap = pw.wav2world(x, fs, frame_period=5.0)
    # print(f0.shape)

    f0_shifted = []
    for j in range(len(unique_note_dur)):
        f0_shifted.extend(new_f0[j] * np.ones(int(unique_note_dur[j] * 200)))
    # check the length of f0_shifted, if not equal to the length of f0, then repeat the last value
    if len(f0_shifted) < len(f0):
        f0_shifted.extend(f0[-(len(f0) - len(f0_shifted)) :])
    f0_shifted = np.array(f0_shifted)
    # print(f0_shifted)
    # get the new audio signal
    y = pw.synthesize(f0_shifted, sp, ap, fs)
    # refine the audio
    y = y.astype(np.float32)
    y = y / np.max(np.abs(y))
    # Save the modified audio signal
    sf.write(f"{tgt_wav_path}/{item_id}.wav", y, fs)
    # convert freq to midi
    midis = []
    for idx in range(len(new_f0)):
        f = new_f0[idx]
        if f == 0:
            midis += [0] * rep_count[idx]
            continue
        midis += [freq_to_midi(f)] * rep_count[idx]
    # print(midis)
    if len(midis) != len(item["notes"]):
        print(len(midis), len(item["notes"]))
        print(midis)
        print(item["notes"])
        print(item["notes_dur"])
        print(rep_count)
        raise ValueError("The length of midis is not equal to the length of notes")
    item["notes"] = midis
    item["item_name"] = f"{singer}#{song}-shift#{item_id}"
    return item


# with open(db4en_json_fn, 'r') as f:
#     data = json.load(f)
# import multiprocessing
# from multiprocessing import Pool
# progress_bar = tqdm(total=len(data), desc='Processing data')
# # Set the number of parallel processes to use
# num_processes = multiprocessing.cpu_count()
# # Create a pool of worker processes
# pool = Pool(processes=num_processes)
# # Map the data processing function to the list of data items
# processed_data = pool.map(process_item, data)
# # Close the pool of worker processes
# pool.close()
# pool.join()
# # Save the processed data to a file
# with open(db4en_shifted_json_fn, 'w') as f:
#     f.write(json.dumps(processed_data, indent=4, ensure_ascii=False))

#     with open(json_fn, 'a', encoding='utf-8') as f:
#     json.dump(info, f, ensure_ascii=False)
#     f.write('\n')
# return info


if __name__ == "__main__":
    # original meta json file
    db4en_json_fn = "data/meta/db4en-wdb.json"
    db4cn_json_fn = "data/meta/db4cn-wdb.json"
    # original wav root path
    db4en_wav_path = "data/mfa_workspace/db4-en/input"
    db4cn_wav_path = "data/mfa_workspace/db4-cn/input"
    # shifted wav root path
    db4en_shift_wav_path = "data/db4-pitchshift/db4#en-shift"
    db4cn_shift_wav_path = "data/db4-pitchshift/db4#cn-shift"
    # shifted meta json file
    db4en_shift_json_fn = "data/meta/db4en-shift-wdb.json"
    db4cn_shift_json_fn = "data/meta/db4cn-shift-wdb.json"

    def run_pitch_shift(flag):
        if flag == "en":
            ori_meta_fn = db4en_json_fn
            ori_wav_path = db4en_wav_path
            tgt_wav_path = db4en_shift_wav_path
            tgt_meta_fn = db4en_shift_json_fn
        elif flag == "cn":
            ori_meta_fn = db4cn_json_fn
            ori_wav_path = db4cn_wav_path
            tgt_wav_path = db4cn_shift_wav_path
            tgt_meta_fn = db4cn_shift_json_fn

        # 如果目标文件存在，则删除文件
        if os.path.exists(tgt_meta_fn):
            os.remove(tgt_meta_fn)
        # 从原始文件中读取并处理item
        with open(ori_meta_fn, "r") as f:
            lines = f.readlines()
        for line in tqdm(lines):
            item = eval(line)
            info = process_item(item, ori_wav_path, tgt_wav_path)
            with open(tgt_meta_fn, "a", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False)
                f.write("\n")

    flags = ["en", "cn"]
    for x in flags:
        run_pitch_shift(x)
