import os
import json
import textgrid
from collections import Counter
from pypinyin import lazy_pinyin
import numpy as np
from functools import reduce
from tqdm import tqdm


# 主要还是根据m4singer原始的tg文件，但是韵母部分根据mfa得到的韵母部分对应的音素成比例划分
####================================ 无用分析function======================
#
# 查看m4singer原始标注和mfa标注的两个textgrid间的关系
def word_info_from_tg(tg_path, ign_sil, which):
    tg = textgrid.TextGrid.fromFile(tg_path)
    word_iter = tg[0]
    word_time_list = []
    print(f"there are {len(word_iter)} word intervals")
    if not ign_sil:
        for i, itv in enumerate(word_iter):
            x = round(itv.maxTime - itv.minTime, 4)
            print(
                f'the number {i}: TEXT={itv.mark if itv.mark not in ["", "<SP>", "<AP>"] else "None"}\tBEFORE={itv.minTime}\tDURATION={x}\tAFTER={itv.maxTime}'
            )
            if which == "BEFORE":
                word_time_list.append(itv.minTime)
            elif which == "DURATION":
                word_time_list.append(x)
            elif which == "AFTER":
                word_time_list.append(itv.maxTime)

    else:
        for i, itv in enumerate(word_iter):
            if itv.mark not in ["", "<SP>", "<AP>"]:  #
                x = round(itv.maxTime - itv.minTime, 4)
                print(
                    f"the number {i}: TEXT={itv.mark}\tBEFORE={itv.minTime}\tDURATION={x}\tAFTER={itv.maxTime}"
                )
                if which == "BEFORE":
                    word_time_list.append(itv.minTime)
                elif which == "DURATION":
                    word_time_list.append(x)
                elif which == "AFTER":
                    word_time_list.append(itv.maxTime)
    print(len(word_time_list), word_time_list)
    return word_time_list


def compare_m4_mfa(singer, song, id, ign_sil=True, which="BEFORE"):
    root_m4_tg = "/Netdata/AudioData/m4singer"
    root_mfa_tg = "data/mfa_workspace/m4singer/output"
    m4_tg_fn = f"{root_m4_tg}/{singer}#{song}/{id}.TextGrid"
    mfa_tg_fn = f"{root_mfa_tg}/{singer}/{song}#{id}.TextGrid"

    time_m4 = word_info_from_tg(m4_tg_fn, ign_sil, which)
    time_mfa = word_info_from_tg(mfa_tg_fn, ign_sil, which)
    print("=====" * 30)
    print(len(time_m4), time_m4)
    print(len(time_mfa), time_mfa)
    print("=====" * 30)


def words_from_tg(tg_fn):
    tg = textgrid.TextGrid.fromFile(tg_fn)
    print(tg)
    word_iter = tg[0]
    words = []
    for itv in word_iter:
        if itv.mark not in ["", "<SP>", "<AP>"]:
            words.append(itv.mark)
    return words


def phone_level(tg_path):
    tg = textgrid.TextGrid.fromFile(tg_path)
    phone_iter = tg[1]
    print(len(phone_iter))


######=============================== 无用分析function========================

############========================检查textgrid文件信息是否被json文件包含===========================
## all about m4singer: textgrid and json files
meta_data = json.load(open("/Netdata/AudioData/m4singer/meta.json", "r"))


# print(type(meta_data))
# print(meta_data[0].keys())
def locate_item(singer, song, id):
    item_name = f"{singer}#{song}#{id}"
    for d in meta_data:
        if d["item_name"] == item_name:
            print(d)
            item_name, phs, txt, is_slur, ph_dur, notes, notes_dur = (
                d["item_name"],
                d["phs"],
                d["txt"],
                d["is_slur"],
                d["ph_dur"],
                d["notes"],
                d["notes_dur"],
            )
            return item_name, phs, txt, is_slur, ph_dur, notes, notes_dur
        else:
            pass


def compare2re_lst(list1, list2):
    # list1 = [1, 2, 2, 3, 4, 4, 5]
    # list2 = [2, 3, 4, 4, 5, 5, 6]
    # 计算两个列表中元素的计数
    counter1 = Counter(list1)
    counter2 = Counter(list2)
    # 获取仅存在于list1中的元素及其重复次数
    only1 = counter1 - counter2
    # 获取仅存在于list2中的元素及其重复次数
    only2 = counter2 - counter1
    print(f"only in 1st list: {only1}")  # 输出: Counter({1: 1})
    print(f"only in 2nd list: {only2}")  # 输出: Counter({6: 1, 5: 1})
    return only1, only2


def check_tg_included_by_json(singer, song, id):
    ## from json file
    item_name, phs, txt, is_slur, ph_dur, notes, notes_dur = locate_item(
        singer, song, id
    )

    ## from tg file
    tg_fn = f"/Netdata/AudioData/m4singer/raw/{singer}#{song}/{id}.TextGrid"
    tg = textgrid.TextGrid.fromFile(tg_fn)
    word_iter = tg[0]
    phone_tier = tg[1]
    phs_in_tg = [itv.mark for itv in phone_tier]

    # whether tg consistent with json file
    # phs_in_tg should be included in phs
    only1, only2 = compare2re_lst(phs_in_tg, phs)
    if len(only1) != 0:
        print(
            f"{phs_in_tg} in textgrid, but not in json file, which is not supposed to be"
        )
        print(len(phs_in_tg), phs_in_tg)
        print(len(phs), phs)
        print(is_slur)


############========================检查textgrid文件信息是否被json文件包含===========================
def word_phone_from_tg(tg_path):
    tg = textgrid.TextGrid.fromFile(tg_path)
    word_tier = tg[0]
    phone_tier = tg[1]
    words, notes = [], []
    word_phone_mapping_list = []  # 第i个元素的值意味着第i个word对应的phone的index
    for i, interval in enumerate(word_tier):
        words.append(interval.mark)
        word_start_time = interval.minTime
        word_end_time = interval.maxTime
        t = []
        phones = []
        phone_dur_list = []
        for j, phone in enumerate(phone_tier):
            phones.append(phone.mark)
            phone_start_time = phone.minTime
            phone_end_time = phone.maxTime
            phone_dur_list.append(round(phone_end_time - phone_start_time, 4))
            if phone_start_time >= word_start_time and phone_end_time <= word_end_time:
                t.append(j)
        word_phone_mapping_list.append(t)
    # print(f'words is {words}')
    # print(f'phones is {phones}')
    # print(f'word_phone_mapping_list is {word_phone_mapping_list}')
    # print(f'phone_dur_list is {phone_dur_list}')
    return words, phones, word_phone_mapping_list, phone_dur_list


# accroding to is_slur, get the connection between json and tg for the same phone
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


def group_wo_slur(re, ph_dur, notes, notes_dur):
    # re = [[0],[1,2],[3,4]]
    # ph_dur = [1,22,333,444,5]
    # ph_dur = [[1], [22, 333], [444, 5]]
    g_ph_dur = []
    g_notes = []
    g_notes_dur = []
    for cur_lst in re:
        g_ph_dur.append([ph_dur[i] for i in cur_lst])
        g_notes.append([notes[i] for i in cur_lst])
        g_notes_dur.append([notes_dur[i] for i in cur_lst])
    return g_ph_dur, g_notes, g_notes_dur


def split_time(mfa_dur_lst, m4_dur_sum):
    mfa_dur = np.array(mfa_dur_lst)
    ratio = mfa_dur / np.sum(mfa_dur)
    dur = np.round(ratio * m4_dur_sum, decimals=4)
    dur[-1] = round(m4_dur_sum - np.sum(dur[:-1]), 4)
    return list(dur)


# 根据mfa中cmu对应的音素时长比例，切割m4原始标注中的韵母时长
# 根据切分后的时长进一步处理is_slur情况中的note
def locate_idx_percent_note(target, note_dur_in_yunmu):
    cur_sum = 0
    for idx, dur in enumerate(note_dur_in_yunmu):
        cur_sum += dur
        if cur_sum >= target:
            percent = 1 - (cur_sum - target) / note_dur_in_yunmu[idx]
            return idx, percent


def examine_wdb_len(new_wdb, new_notes):
    wdb = (lambda x, y: x + y, new_wdb)
    notes = (lambda x, y: x + y, new_notes)
    if len(wdb) != len(notes):
        print("=========MISTAKE IN EXAMINE WDB_LENGTH========")
        print(new_wdb, new_notes)
        print(f"wdb = {wdb}\n notes = {notes}")


def convert(
    item_name,
    word_m4_mfa_lst,
    m4_word_ph_lst,
    m4_phones_lst,
    m4_ph_dur_lst,
    mfa_word_ph_lst,
    mfa_phones_lst,
    mfa_ph_dur_lst,
    g_ph_dur,
    g_notes,
    g_notes_dur,
):
    # print('ENTER DEBUG_CONVERT')
    # print(f'word_m4_mfa_lst \n \
    #       m4_word_ph_lst, m4_phones_lst, m4_ph_dur_lst, \n \
    #       mfa_word_ph_lst, mfa_phones_lst, mfa_ph_dur_lst, \n \
    #       g_ph_dur, g_notes, g_notes_dur')
    # print(f'==================m4_ph_dur_lst={m4_ph_dur_lst}===================')
    # print(f'===================g_ph_dur={g_ph_dur}===================')
    # print(f'===================g_notes={g_notes}===================')
    # print(f'===================g_notes_dur={g_notes_dur}===================')
    # for i in (word_m4_mfa_lst, m4_word_ph_lst, m4_phones_lst, m4_ph_dur_lst, mfa_word_ph_lst, mfa_phones_lst, mfa_ph_dur_lst, g_ph_dur, g_notes, g_notes_dur):
    #     print(len(i), i)
    #     print('==='*30)
    new_phs = []
    new_ph_dur = []
    new_notes = []
    new_notes_dur = []
    new_is_slur = []
    new_wdb = []
    examine_wdb_len(new_wdb, new_phs)
    for m4_idx, mfa_idx in enumerate(word_m4_mfa_lst):
        m4_ph_idx = m4_word_ph_lst[m4_idx]
        mfa_ph_idx = mfa_word_ph_lst[mfa_idx] if mfa_idx != None else None
        # print('===='*10)
        if mfa_idx == None:  # <AP> or <SP>
            # import pdb; pdb.set_trace()
            new_phs.append([m4_phones_lst[m4_ph_idx[0]]])
            # new_ph_dur.append([m4_ph_dur_lst[m4_ph_idx[0]]])
            new_ph_dur.append(g_ph_dur[m4_ph_idx[0]])
            new_notes.append(g_notes[m4_ph_idx[0]])
            new_notes_dur.append(g_notes_dur[m4_ph_idx[0]])
            new_is_slur.append([0])
            if m4_idx == 0:  # 句子开始的<AP> or <SP>不作为word boundary
                new_wdb.append([0])
            else:  # 除了句子开始的<AP> or <SP>，其他位置的都是word boundary
                new_wdb.append([1])
            examine_wdb_len(new_wdb, new_phs)
        else:
            # print(f'{m4_ph_idx}\t{mfa_ph_idx}')
            m4_ph = [m4_phones_lst[i] for i in m4_ph_idx]
            mfa_ph = [mfa_phones_lst[i] for i in mfa_ph_idx]
            # print(f'{m4_ph}\t{mfa_ph}')
            m4_ph_dur = [m4_ph_dur_lst[i] for i in m4_ph_idx]
            mfa_ph_dur = [mfa_ph_dur_lst[i] for i in mfa_ph_idx]
            # print(f'{m4_ph_dur}\t{mfa_ph_dur}')

            t_g_ph_dur = [
                g_ph_dur[i] for i in m4_ph_idx
            ]  # 当前的pinyin phone在原始json标注里对应的phone dur
            t_g_notes = [g_notes[i] for i in m4_ph_idx]
            t_g_notes_dur = [g_notes_dur[i] for i in m4_ph_idx]
            # print(f'{t_g_ph_dur}\t{t_g_notes}\t{t_g_notes_dur}')
            # [9, 10]	[13, 14]
            # ['sh', 'i']	['SH', 'IY']
            # [0.22, 0.29]	[0.2, 0.24]
            # [[0.22], [0.1091, 0.1809]]	[[64], [64, 65]]	[[0.3291], [0.3291, 0.1809]]

            # 1.声母+韵母
            if len(m4_ph) == 2:
                # 1.1 pinyin里单个声母对应cmu里单个音素
                if m4_ph[0] != "c":  # 是非c的声母，只有c这个声母被分为了两个cmu音素
                    ## 1.1.1声母部分
                    new_phs.append([mfa_ph[0]])
                    new_ph_dur.append(t_g_ph_dur[0])
                    new_notes.append(t_g_notes[0])
                    new_notes_dur.append(t_g_notes_dur[0])
                    new_is_slur.append([0])
                    new_wdb.append([0])
                    examine_wdb_len(new_wdb, new_phs)
                    ## 韵母部分
                    num_notes_in_yunmu = len(t_g_notes[-1])
                    # yunmu2cmu_durs = split_time(mfa_ph_dur[1:], m4_ph_dur[-1])
                    yunmu2cmu_durs = split_time(mfa_ph_dur[1:], sum(t_g_ph_dur[-1]))
                    num_yunmu2cmu = len(mfa_ph_dur[1:])
                    ### 1.1.2没有连音，一个韵母只对应一个note
                    if num_notes_in_yunmu == 1:
                        new_phs.append(mfa_ph[1:])
                        new_ph_dur.append(yunmu2cmu_durs)
                        new_notes.append(t_g_notes[-1] * num_yunmu2cmu)
                        new_notes_dur.append(t_g_notes_dur[-1] * num_yunmu2cmu)
                        new_is_slur.append([0] * num_yunmu2cmu)
                        new_wdb.append([0] * (num_yunmu2cmu - 1) + [1])
                        examine_wdb_len(new_wdb, new_phs)
                    ### 1.1.3有连音，一个韵母对应多个note
                    else:
                        # 30 {'a': ['AA'], 'ai': ['AY'], 'ao': ['AW'], 'b': ['B'], 'ch': ['CH'], 'd': ['D'], 'e': ['ER'], 'ei': ['EY'], 'f': ['F'], 'g': ['G'], 'h': ['HH'], 'i': ['IY'], 'j': ['J'], 'k': ['K'], 'l': ['L'], 'm': ['M'], 'n': ['N'], 'o': ['AO'], 'ou': ['OW'], 'p': ['P'], 'q': ['Q'], 'r': ['R'], 'sh': ['SH'], 's': ['S'], 't': ['T'], 'u': ['UW'], 'x': ['X'], 'zh': ['JH'], 'z': ['Z'], 'y': ['Y']}
                        # 19 {'an': ['AE', 'N'], 'ang': ['AE', 'NG'], 'c': ['T', 'S'], 'en': ['AH', 'N'], 'eng': ['AH', 'NG'], 'er': ['AA', 'R'], 'ia': ['IY', 'AA'], 'iao': ['IY', 'AW'], 'ie': ['IY', 'EH'], 'ing': ['IY', 'NG'], 'in': ['IY', 'N'], 'ong': ['UH', 'NG'], 'uai': ['UW', 'AY'], 'ua': ['UW', 'AA'], 'uo': ['UW', 'AO'], 've': ['IY', 'EH'], 'v': ['IY', 'UW'], 'iou': ['IY', 'UH'], 'uei': ['UW', 'IY']}
                        # 8 {'iang': ['IY', 'AE', 'NG'], 'ian': ['IY', 'AE', 'N'], 'iong': ['IY', 'UH', 'NG'], 'uang': ['UW', 'AE', 'NG'], 'uan': ['UW', 'AE', 'N'], 'vn': ['UW', 'AH', 'N'], 'uen': ['UW', 'AH', 'N'], 'van': ['UW', 'AE', 'N']}
                        if num_yunmu2cmu == 1:  # 1.1.3.1，一个韵母对应1个cmu音素
                            new_phs.append([mfa_ph[1]] * num_notes_in_yunmu)
                            new_ph_dur.append(t_g_ph_dur[-1])
                            new_notes.append(t_g_notes[-1])
                            new_notes_dur.append(t_g_notes_dur[-1])
                            new_is_slur.append([0] + [1] * (num_notes_in_yunmu - 1))
                            new_wdb.append([0] * (num_notes_in_yunmu - 1) + [1])
                            examine_wdb_len(new_wdb, new_phs)
                        elif num_yunmu2cmu == 2:  # 1.1.3.2，一个韵母对应2个cmu音素
                            # note_dur_in_yunmu = t_g_ph_dur[-1] # 一个韵母被不同note占据的时长
                            idx, percent = locate_idx_percent_note(
                                yunmu2cmu_durs[0], t_g_ph_dur[-1]
                            )
                            # if idx == 0:    #如果落在第一个note的区间内，将第一个note直接给第一个cmu音素，剩下的都给第二个音素
                            #     dur1 = [t_g_ph_dur[-1][0]]
                            #     dur2 = t_g_ph_dur[-1][1:]
                            # elif idx == num_notes_in_yunmu-1:    #如果落在最后一个note的区间内，将最后一个note直接给第二个cmu音素，剩下的都给第一个音素
                            #     dur1 = t_g_ph_dur[-1][:-1]
                            #     dur2 = [t_g_ph_dur[-1][-1]]
                            # elif percent<0.3:   #不进行分割，直接把locate的这个note分给第二个cmu音素
                            #     dur1 = t_g_ph_dur[-1][:idx]
                            #     dur2 = t_g_ph_dur[-1][idx:]
                            # elif 0.3<percent<0.7:   #进行分割，将locate的这个note分别分给第一个、第二个cmu音素
                            #     part1 = round(t_g_ph_dur[-1][idx]*percent, 4)
                            #     part2 = round(t_g_ph_dur[-1][idx]-part1, 4)
                            #     dur1 = t_g_ph_dur[-1][:idx]+[part1]
                            #     dur2 = [part2] + t_g_ph_dur[-1][idx+1:]
                            # elif percent>0.7:   #不进行分割，直接把locate的这个note分给第一个cmu音素
                            #     dur1 = t_g_ph_dur[-1][:idx+1]
                            #     dur2 = t_g_ph_dur[-1][idx+1:]
                            part1 = round(t_g_ph_dur[-1][idx] * percent, 4)
                            part2 = round(t_g_ph_dur[-1][idx] - part1, 4)
                            dur1 = t_g_ph_dur[-1][:idx] + [part1]
                            dur2 = [part2] + t_g_ph_dur[-1][idx + 1 :]
                            # ============
                            new_phs.append(
                                [mfa_ph[1]] * len(dur1) + [mfa_ph[2]] * len(dur2)
                            )
                            new_ph_dur.append(dur1 + dur2)
                            new_notes.append(
                                t_g_notes[-1][: len(dur1)] + t_g_notes[-1][-len(dur2) :]
                            )
                            new_notes_dur.append(
                                t_g_notes_dur[-1][: len(dur1)]
                                + t_g_notes_dur[-1][-len(dur2) :]
                            )
                            new_is_slur.append(
                                [0]
                                + [1] * ((len(dur1) - 1))
                                + [0]
                                + [1] * ((len(dur2) - 1))
                            )
                            new_wdb.append([0] * ((len(dur1 + dur2)) - 1) + [1])
                            examine_wdb_len(new_wdb, new_phs)
                        elif num_yunmu2cmu == 3:  # 1.1.3.3，一个韵母对应3个cmu音素
                            idx12, percent12 = locate_idx_percent_note(
                                yunmu2cmu_durs[0], t_g_ph_dur[-1]
                            )
                            idx23, percent23 = locate_idx_percent_note(
                                sum(yunmu2cmu_durs[:2]), t_g_ph_dur[-1]
                            )
                            part12 = round(t_g_ph_dur[-1][idx12] * percent12, 4)
                            part21 = round(t_g_ph_dur[-1][idx12] - part12, 4)
                            part23 = round(t_g_ph_dur[-1][idx23] * percent23, 4)
                            part32 = round(t_g_ph_dur[-1][idx23] - part23, 4)
                            dur1 = t_g_ph_dur[-1][:idx12] + [part12]
                            dur3 = [part32] + t_g_ph_dur[-1][1 + idx23 :]
                            if idx12 == idx23:  # 如果落在同一个note区间
                                dur2 = [
                                    round(t_g_ph_dur[-1][idx12] - part12 - part32, 4)
                                ]
                            else:
                                dur2 = (
                                    [part21]
                                    + t_g_ph_dur[-1][
                                        1 + idx12 : -(num_notes_in_yunmu - idx23)
                                    ]
                                    + [part23]
                                )
                            new_phs.append(
                                [mfa_ph[1]] * len(dur1)
                                + [mfa_ph[2]] * len(dur2)
                                + [mfa_ph[3]] * len(dur3)
                            )
                            new_ph_dur.append(dur1 + dur2 + dur3)
                            new_notes.append(
                                t_g_notes[-1][: len(dur1)]
                                + t_g_notes[-1][idx12 : idx12 + len(dur2)]
                                + t_g_notes[-1][-len(dur3) :]
                            )
                            new_notes_dur.append(
                                t_g_notes_dur[-1][: len(dur1)]
                                + t_g_notes_dur[-1][idx12 : idx12 + len(dur2)]
                                + t_g_notes_dur[-1][-len(dur3) :]
                            )
                            new_is_slur.append(
                                [0]
                                + [1] * ((len(dur1) - 1))
                                + [0]
                                + [1] * ((len(dur2) - 1))
                                + [0]
                                + [1] * ((len(dur3) - 1))
                            )
                            new_wdb.append([0] * (len(dur1 + dur2 + dur3) - 1) + [1])
                            examine_wdb_len(new_wdb, new_phs)
                else:  # 处理声母c对应的汉字
                    # [3, 4]	[3, 4, 5]
                    # ['c', 'ai']	['T', 'S', 'AY']
                    # [0.25, 0.4]	[0.08, 0.15, 0.44]
                    # [[0.25], [0.1891, 0.2109]]	[[64], [64, 65]]	[[0.4391], [0.4391, 0.2109]]
                    # dur_ts = split_time(mfa_ph_dur[:2], m4_ph_dur[0])
                    dur_ts = split_time(mfa_ph_dur[:2], sum(t_g_ph_dur[0]))
                    # dur_t, dur_s = dur_ts[0], dur_ts[1]
                    ## 1.1.1声母部分
                    new_phs.append(mfa_ph[:2])
                    new_ph_dur.append(dur_ts)
                    new_notes.append(t_g_notes[0] * 2)
                    new_notes_dur.append(t_g_notes_dur[0] * 2)
                    new_is_slur.append([0] * 2)
                    new_wdb.append([0] * 2)
                    examine_wdb_len(new_wdb, new_phs)
                    ## 韵母部分
                    num_notes_in_yunmu = len(t_g_notes[-1])
                    # yunmu2cmu_durs = split_time(mfa_ph_dur[2:], m4_ph_dur[-1])
                    yunmu2cmu_durs = split_time(mfa_ph_dur[2:], sum(t_g_ph_dur[-1]))
                    num_yunmu2cmu = len(mfa_ph_dur[2:])
                    ### 1.1.2没有连音，一个韵母只对应一个note
                    if num_notes_in_yunmu == 1:
                        new_phs.append(mfa_ph[2:])
                        new_ph_dur.append(yunmu2cmu_durs)
                        new_notes.append(t_g_notes[-1] * num_yunmu2cmu)
                        new_notes_dur.append(t_g_notes_dur[-1] * num_yunmu2cmu)
                        new_is_slur.append([0] * num_yunmu2cmu)
                        new_wdb.append([0] * (num_yunmu2cmu - 1) + [1])
                        examine_wdb_len(new_wdb, new_phs)
                    ### 1.1.3有连音，一个韵母对应多个note
                    else:
                        # 30 {'a': ['AA'], 'ai': ['AY'], 'ao': ['AW'], 'b': ['B'], 'ch': ['CH'], 'd': ['D'], 'e': ['ER'], 'ei': ['EY'], 'f': ['F'], 'g': ['G'], 'h': ['HH'], 'i': ['IY'], 'j': ['J'], 'k': ['K'], 'l': ['L'], 'm': ['M'], 'n': ['N'], 'o': ['AO'], 'ou': ['OW'], 'p': ['P'], 'q': ['Q'], 'r': ['R'], 'sh': ['SH'], 's': ['S'], 't': ['T'], 'u': ['UW'], 'x': ['X'], 'zh': ['JH'], 'z': ['Z'], 'y': ['Y']}
                        # 19 {'an': ['AE', 'N'], 'ang': ['AE', 'NG'], 'c': ['T', 'S'], 'en': ['AH', 'N'], 'eng': ['AH', 'NG'], 'er': ['AA', 'R'], 'ia': ['IY', 'AA'], 'iao': ['IY', 'AW'], 'ie': ['IY', 'EH'], 'ing': ['IY', 'NG'], 'in': ['IY', 'N'], 'ong': ['UH', 'NG'], 'uai': ['UW', 'AY'], 'ua': ['UW', 'AA'], 'uo': ['UW', 'AO'], 've': ['IY', 'EH'], 'v': ['IY', 'UW'], 'iou': ['IY', 'UH'], 'uei': ['UW', 'IY']}
                        # 8 {'iang': ['IY', 'AE', 'NG'], 'ian': ['IY', 'AE', 'N'], 'iong': ['IY', 'UH', 'NG'], 'uang': ['UW', 'AE', 'NG'], 'uan': ['UW', 'AE', 'N'], 'vn': ['UW', 'AH', 'N'], 'uen': ['UW', 'AH', 'N'], 'van': ['UW', 'AE', 'N']}
                        if num_yunmu2cmu == 1:  # 1.1.3.1，一个韵母对应1个cmu音素
                            new_phs.append([mfa_ph[2]] * num_notes_in_yunmu)
                            new_ph_dur.append(t_g_ph_dur[-1])
                            new_notes.append(t_g_notes[-1])
                            new_notes_dur.append(t_g_notes_dur[-1])
                            new_is_slur.append([0] + [1] * (num_notes_in_yunmu - 1))
                            new_wdb.append([0] * (num_notes_in_yunmu - 1) + [1])
                            examine_wdb_len(new_wdb, new_phs)
                        elif num_yunmu2cmu == 2:  # 1.1.3.2，一个韵母对应2个cmu音素
                            # note_dur_in_yunmu = t_g_ph_dur[-1] # 一个韵母被不同note占据的时长
                            idx, percent = locate_idx_percent_note(
                                yunmu2cmu_durs[0], t_g_ph_dur[-1]
                            )
                            part1 = round(t_g_ph_dur[-1][idx] * percent, 4)
                            part2 = round(t_g_ph_dur[-1][idx] - part1, 4)
                            dur1 = t_g_ph_dur[-1][:idx] + [part1]
                            dur2 = [part2] + t_g_ph_dur[-1][idx + 1 :]
                            # ============
                            new_phs.append(
                                [mfa_ph[2]] * len(dur1) + [mfa_ph[3]] * len(dur2)
                            )
                            new_ph_dur.append(dur1 + dur2)
                            new_notes.append(
                                t_g_notes[-1][: len(dur1)] + t_g_notes[-1][-len(dur2) :]
                            )
                            new_notes_dur.append(
                                t_g_notes_dur[-1][: len(dur1)]
                                + t_g_notes_dur[-1][-len(dur2) :]
                            )
                            new_is_slur.append(
                                [0]
                                + [1] * ((len(dur1) - 1))
                                + [0]
                                + [1] * ((len(dur2) - 1))
                            )
                            new_wdb.append([0] * (len(dur1 + dur2) - 1) + [1])
                            examine_wdb_len(new_wdb, new_phs)
                        elif num_yunmu2cmu == 3:  # 1.1.3.3，一个韵母对应3个cmu音素
                            idx12, percent12 = locate_idx_percent_note(
                                yunmu2cmu_durs[0], t_g_ph_dur[-1]
                            )
                            idx23, percent23 = locate_idx_percent_note(
                                sum(yunmu2cmu_durs[:2]), t_g_ph_dur[-1]
                            )
                            part12 = round(t_g_ph_dur[-1][idx12] * percent12, 4)
                            part21 = round(t_g_ph_dur[-1][idx12] - part12, 4)
                            part23 = round(t_g_ph_dur[-1][idx23] * percent23, 4)
                            part32 = round(t_g_ph_dur[-1][idx23] - part23, 4)
                            dur1 = t_g_ph_dur[-1][:idx12] + [part12]
                            dur3 = [part32] + t_g_ph_dur[-1][1 + idx23 :]
                            if idx12 == idx23:  # 如果落在同一个note区间
                                dur2 = [
                                    round(t_g_ph_dur[-1][idx12] - part12 - part32, 4)
                                ]
                            else:
                                dur2 = (
                                    [part21]
                                    + t_g_ph_dur[-1][
                                        1 + idx12 : -(num_notes_in_yunmu - idx23)
                                    ]
                                    + [part23]
                                )
                            new_phs.append(
                                [mfa_ph[2]] * len(dur1)
                                + [mfa_ph[3]] * len(dur2)
                                + [mfa_ph[4]] * len(dur3)
                            )
                            new_ph_dur.append(dur1 + dur2 + dur3)
                            new_notes.append(
                                t_g_notes[-1][: len(dur1)]
                                + t_g_notes[-1][idx12 : idx12 + len(dur2)]
                                + t_g_notes[-1][-len(dur3) :]
                            )
                            new_notes_dur.append(
                                t_g_notes_dur[-1][: len(dur1)]
                                + t_g_notes_dur[-1][idx12 : idx12 + len(dur2)]
                                + t_g_notes_dur[-1][-len(dur3) :]
                            )
                            new_is_slur.append(
                                [0]
                                + [1] * ((len(dur1) - 1))
                                + [0]
                                + [1] * ((len(dur2) - 1))
                                + [0]
                                + [1] * ((len(dur3) - 1))
                            )
                            new_wdb.append([0] * (len(dur1 + dur2 + dur3) - 1) + [1])
                            examine_wdb_len(new_wdb, new_phs)

            else:  # 无声母
                # [17]	[28, 29, 30]
                # ['iang']	['Y', 'AE', 'NG']
                # [0.34]	[0.09, 0.14, 0.03]
                # [[0.34]]	[[63]]	[[0.34]]
                num_notes_in_yunmu = len(t_g_notes[-1])
                # yunmu2cmu_durs = split_time(mfa_ph_dur, m4_ph_dur[-1])
                yunmu2cmu_durs = split_time(mfa_ph_dur, sum(t_g_ph_dur[-1]))
                num_yunmu2cmu = len(mfa_ph_dur)
                ### 1.1.2没有连音，一个韵母只对应一个note
                if num_notes_in_yunmu == 1:
                    new_phs.append(mfa_ph)
                    new_ph_dur.append(yunmu2cmu_durs)
                    new_notes.append(t_g_notes[-1] * num_yunmu2cmu)
                    new_notes_dur.append(t_g_notes_dur[-1] * num_yunmu2cmu)
                    new_is_slur.append([0] * num_yunmu2cmu)
                    new_wdb.append([0] * (num_yunmu2cmu - 1) + [1])
                    examine_wdb_len(new_wdb, new_phs)
                ### 1.1.3有连音，一个韵母对应多个note
                else:
                    if num_yunmu2cmu == 1:  # 1.1.3.1，一个韵母对应1个cmu音素
                        new_phs.append([mfa_ph[0]] * num_notes_in_yunmu)
                        new_ph_dur.append(t_g_ph_dur[-1])
                        new_notes.append(t_g_notes[-1])
                        new_notes_dur.append(t_g_notes_dur[-1])
                        new_is_slur.append([0] + [1] * (num_notes_in_yunmu - 1))
                        new_wdb.append([0] * (num_notes_in_yunmu - 1) + [1])
                        examine_wdb_len(new_wdb, new_phs)
                    elif num_yunmu2cmu == 2:  # 1.1.3.2，一个韵母对应2个cmu音素
                        # note_dur_in_yunmu = t_g_ph_dur[-1] # 一个韵母被不同note占据的时长
                        idx, percent = locate_idx_percent_note(
                            yunmu2cmu_durs[0], t_g_ph_dur[-1]
                        )
                        part1 = round(t_g_ph_dur[-1][idx] * percent, 4)
                        part2 = round(t_g_ph_dur[-1][idx] - part1, 4)
                        dur1 = t_g_ph_dur[-1][:idx] + [part1]
                        dur2 = [part2] + t_g_ph_dur[-1][idx + 1 :]
                        # ============
                        new_phs.append(
                            [mfa_ph[0]] * len(dur1) + [mfa_ph[1]] * len(dur2)
                        )
                        new_ph_dur.append(dur1 + dur2)
                        new_notes.append(
                            t_g_notes[-1][: len(dur1)] + t_g_notes[-1][-len(dur2) :]
                        )
                        new_notes_dur.append(
                            t_g_notes_dur[-1][: len(dur1)]
                            + t_g_notes_dur[-1][-len(dur2) :]
                        )
                        new_is_slur.append(
                            [0]
                            + [1] * ((len(dur1) - 1))
                            + [0]
                            + [1] * ((len(dur2) - 1))
                        )
                        new_wdb.append([0] * (len(dur1 + dur2) - 1) + [1])
                        examine_wdb_len(new_wdb, new_phs)
                    elif num_yunmu2cmu == 3:  # 1.1.3.3，一个韵母对应3个cmu音素
                        idx12, percent12 = locate_idx_percent_note(
                            yunmu2cmu_durs[0], t_g_ph_dur[-1]
                        )
                        idx23, percent23 = locate_idx_percent_note(
                            sum(yunmu2cmu_durs[:2]), t_g_ph_dur[-1]
                        )
                        part12 = round(t_g_ph_dur[-1][idx12] * percent12, 4)
                        part21 = round(t_g_ph_dur[-1][idx12] - part12, 4)
                        part23 = round(t_g_ph_dur[-1][idx23] * percent23, 4)
                        part32 = round(t_g_ph_dur[-1][idx23] - part23, 4)
                        # import pdb; pdb.set_trace()
                        dur1 = t_g_ph_dur[-1][:idx12] + [part12]
                        dur3 = [part32] + t_g_ph_dur[-1][1 + idx23 :]
                        if idx12 == idx23:  # 如果落在同一个note区间
                            dur2 = [round(t_g_ph_dur[-1][idx12] - part12 - part32, 4)]
                        else:
                            dur2 = (
                                [part21]
                                + t_g_ph_dur[-1][
                                    1 + idx12 : -(num_notes_in_yunmu - idx23)
                                ]
                                + [part23]
                            )
                        new_phs.append(
                            [mfa_ph[0]] * len(dur1)
                            + [mfa_ph[1]] * len(dur2)
                            + [mfa_ph[2]] * len(dur3)
                        )
                        new_ph_dur.append(dur1 + dur2 + dur3)
                        new_notes.append(
                            t_g_notes[-1][: len(dur1)]
                            + t_g_notes[-1][idx12 : idx12 + len(dur2)]
                            + t_g_notes[-1][-len(dur3) :]
                        )
                        new_notes_dur.append(
                            t_g_notes_dur[-1][: len(dur1)]
                            + t_g_notes_dur[-1][idx12 : idx12 + len(dur2)]
                            + t_g_notes_dur[-1][-len(dur3) :]
                        )
                        new_is_slur.append(
                            [0]
                            + [1] * ((len(dur1) - 1))
                            + [0]
                            + [1] * ((len(dur2) - 1))
                            + [0]
                            + [1] * ((len(dur3) - 1))
                        )
                        new_wdb.append([0] * (len(dur1 + dur2 + dur3) - 1) + [1])
                        examine_wdb_len(new_wdb, new_phs)

    # print(f'<===={new_phs}\n{new_ph_dur}\n{new_notes}\n{new_notes_dur}\n{new_is_slur}\n{new_wdb}====>')
    assert (
        len(new_phs)
        == len(new_ph_dur)
        == len(new_notes)
        == len(new_notes_dur)
        == len(new_is_slur)
        == len(new_wdb)
    ), (item_name, new_phs, new_ph_dur, new_notes, new_notes_dur, new_is_slur, new_wdb)
    return new_phs, new_ph_dur, new_notes, new_notes_dur, new_is_slur, new_wdb


def word_align_from_tgs(m4_tg_fn, mfa_tg_fn):
    m4_tg = textgrid.TextGrid.fromFile(m4_tg_fn)
    m4_word_iter = m4_tg[0]
    m4_marks = []
    m4_words = []
    m4_idxes = []
    for idx, itv in enumerate(m4_word_iter):
        m4_marks.append(itv.mark)
        m4_idxes.append(idx)
    for itv in m4_word_iter:
        if itv.mark not in ["", "<SP>", "<AP>"]:
            m4_words.append(itv.mark)

    mfa_tg = textgrid.TextGrid.fromFile(mfa_tg_fn)
    mfa_word_iter = mfa_tg[0]
    mfa_words = []
    mfa_idxes = []
    for idx, itv in enumerate(mfa_word_iter):
        if itv.mark not in ["", "<SP>", "<AP>"]:
            mfa_words.append(itv.mark)
            mfa_idxes.append(idx)
    # print(f'{len(m4_marks)}, {m4_marks}\n{len(m4_words)}, {m4_words}\n{len(mfa_words)}, {mfa_words}')
    assert len(m4_words) == len(
        mfa_words
    ), f"{len(m4_words)} in m4-tg, {len(mfa_words)} in mfa-tg"
    # 17, ['<AP>', '<SP>', '若', '有', '缘', '<SP>', '<AP>', '有', '<SP>', '缘', '就', '能', '期', '待', '明', '天', '<SP>']
    # 11, ['若', '有', '缘', '有', '缘', '就', '能', '期', '待', '明', '天']
    # 11, ['ruo', 'you', 'yuan', 'you', 'yuan', 'jiu', 'neng', 'qi', 'dai', 'ming', 'tian']
    word_m4_mfa_lst = (
        []
    )  # word_m4_mfa_lst[i] means the i word in m4-word-iter-interval ==> the word_m4_mfa_lst[i] word in mfa-word-iter-interval
    m4_point = 0
    for i, mark in enumerate(m4_marks):
        # print(f'{i}==={m4_point}')
        if m4_point == len(mfa_words):
            word_m4_mfa_lst.append(None)
        elif mark == m4_words[m4_point]:
            word_m4_mfa_lst.append(mfa_idxes[m4_point])
            m4_point += 1
        else:
            word_m4_mfa_lst.append(None)
    # [None, None, 1, 2, 4, None, None, 6, None, 8, 9, 10, 11, 12, 13, 14, None]
    return word_m4_mfa_lst


def pipeline(ori_meta_fn, tgt_meta_fn):
    # 如果目标文件存在，则删除
    if os.path.exists(tgt_meta_fn):
        os.remove(tgt_meta_fn)
    with open(ori_meta_fn, "r") as f:
        ori_datas = f.readlines()
    #    ori_datas = json.load(f)
    for d in tqdm(ori_datas):
        d = eval(d)
        # 0. 从m4singer原始textgrid和mfa得到的textgrid分别读取信息
        item_name, phs, txt, is_slur, ph_dur, notes, notes_dur = (
            d["item_name"],
            d["phs"],
            d["txt"],
            d["is_slur"],
            d["ph_dur"],
            d["notes"],
            d["notes_dur"],
        )
        singer, song, id = item_name.split("#")
        m4_tg_fn = f"{root_m4_tg}/{singer}#{song}/{id}.TextGrid"
        mfa_tg_fn = f"{root_mfa_tg}/{singer}/{song}#{id}.TextGrid"
        m4_words_lst, m4_phones_lst, m4_word_ph_lst, m4_ph_dur_lst = word_phone_from_tg(
            m4_tg_fn
        )
        mfa_words_lst, mfa_phones_lst, mfa_word_ph_lst, mfa_ph_dur_lst = (
            word_phone_from_tg(mfa_tg_fn)
        )

        # 1. accroding to the same word in m4-tg and mfa-tg, get word-phone-mappings respectively
        word_m4_mfa_lst = word_align_from_tgs(m4_tg_fn, mfa_tg_fn)

        # 2. 从json file里得到以phone为单位的slur和note相关信息
        # slur = [0,1,0,0,0,1,1,0,0,0,1,1,1,1,0,0]
        # # re = [[0,1],[2],[3],[4,5,6],[7],[8],[9,10,11,12,13],[14],[15]]
        # re = slur_json_tg(slur)
        # print(re)
        re = slur_json_tg(is_slur)
        # print(len(re),re)
        g_ph_dur, g_notes, g_notes_dur = group_wo_slur(re, ph_dur, notes, notes_dur)
        # print(len(g_ph_dur),g_ph_dur)
        # print(len(g_notes),g_notes)
        # print(len(g_notes_dur),g_notes_dur)
        # ### 2.note：检查m4singer中原始标注里，meta.json是否比对应的textgrid多is_slur相关信息，
        # ### 2.note：即meta.json文件里的音素要绝对大于textgrid里的音素数量
        # m4_tg = textgrid.TextGrid.fromFile(m4_tg_fn)
        # phone_tier = m4_tg[1]
        # phs_in_tg = [itv.mark for itv in phone_tier]
        # assert len(phs_in_tg)<=len(phs), f'len(phs_in_tg)={len(phs_in_tg)}, len(phs)={len(phs)}'

        # try:
        # 3. 调用convert函数，根据前面m4-mfa两个textgrid，以及m4的textgrid和meta.json文件里的信息，得到新的以原始pinyin phone为单位的信息
        new_phs, new_ph_dur, new_notes, new_notes_dur, new_is_slur, new_wdb = convert(
            item_name,
            word_m4_mfa_lst,
            m4_word_ph_lst,
            m4_phones_lst,
            m4_ph_dur_lst,
            mfa_word_ph_lst,
            mfa_phones_lst,
            mfa_ph_dur_lst,
            g_ph_dur,
            g_notes,
            g_notes_dur,
        )
        phs = reduce(lambda x, y: x + y, new_phs)
        is_slur = reduce(lambda x, y: x + y, new_is_slur)
        ph_dur = reduce(lambda x, y: x + y, new_ph_dur)
        notes = reduce(lambda x, y: x + y, new_notes)
        notes_dur = reduce(lambda x, y: x + y, new_notes_dur)
        word_boundary = reduce(lambda x, y: x + y, new_wdb)
        assert (
            len(phs)
            == len(is_slur)
            == len(ph_dur)
            == len(notes)
            == len(notes_dur)
            == len(word_boundary)
        ), (
            item_name,
            phs,
            is_slur,
            ph_dur,
            notes,
            notes_dur,
            word_boundary,
            new_phs,
            new_ph_dur,
            new_notes,
            new_notes_dur,
            new_is_slur,
            new_wdb,
        )

        info = {
            "lang": 1,
            "item_name": item_name,
            "txt": txt,
            "words": " ".join(lazy_pinyin(txt)),  # not read from aligned textgrid
            "phs": phs,
            "is_slur": is_slur,
            "ph_dur": ph_dur,
            "notes": notes,
            "notes_dur": notes_dur,
            "word_boundary": word_boundary,
        }

        with open(tgt_meta_fn, "a", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False)
            f.write("\n")
        # except:
        #     print(f'PROBELM when processing {item_name}!')


if __name__ == "__main__":
    root_m4_tg = "/Netdata/AudioData/m4singer"
    root_mfa_tg = "data/mfa_workspace/m4singer/output"
    ori_meta_fn = "data/meta/m4-adjust-ori.json"
    tgt_meta_fn = "data/meta/m4-proportional.json"
    pipeline(ori_meta_fn, tgt_meta_fn)
