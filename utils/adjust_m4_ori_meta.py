import os
import json
import textgrid
from collections import Counter
from tqdm import tqdm


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
    # print(f'only in 1st list: {only1}')  # 输出: Counter({1: 1})
    # print(f'only in 2nd list: {only2}')  # 输出: Counter({6: 1, 5: 1})
    return only1, only2


def get_new_tg(json_lst, tg_lst, is_slur, item_name):
    # 根据json_lst，对tg_lst中的连音部分的pinyin音素进行重复，得到完整的new_tg_lst
    # 会修改tg_lst，所以尽量用copy之后的tg_lst传入函数
    idx_json = 0
    idx_tg = 0
    # print(json_lst)
    while idx_json < len(json_lst):
        # print(f'{idx_json}==={idx_tg}')
        # print(tg_lst)
        if is_slur[idx_json] == 1:
            tg_lst.insert(idx_tg, json_lst[idx_json])
            idx_json += 1
            idx_tg += 1
        elif json_lst[idx_json] == tg_lst[idx_tg]:
            idx_json += 1
            idx_tg += 1
        elif tg_lst[idx_tg - 1] == json_lst[idx_json] and is_slur[idx_json] == 1:
            tg_lst.insert(idx_tg, json_lst[idx_json])
            idx_tg += 1
            idx_json += 1
        elif tg_lst[idx_tg] in ["<AP>", "<SP>"]:
            idx_tg += 1
        else:
            assert False, f"{(item_name, idx_json, idx_tg, json_lst, tg_lst, is_slur)}"
    return tg_lst


def get_idx_sil(json_lst, new_tg_lst):
    # 根据只有ap和sp不同的json_lst和new_tg_lst，得到需要补充的json_lst的下标索引
    idx_sil_dict = {}
    idx_json = 0
    idx_new_tg = 0
    while idx_json < len(json_lst):
        if json_lst[idx_json] == new_tg_lst[idx_new_tg]:
            idx_json += 1
            idx_new_tg += 1
        else:
            idx_sil_dict[f"{idx_new_tg}"] = new_tg_lst[idx_new_tg]
            idx_new_tg += 1
    while idx_new_tg < len(new_tg_lst):
        idx_sil_dict[f"{idx_new_tg}"] = new_tg_lst[idx_new_tg]
        idx_new_tg += 1
    # print(idx_sil_dict)
    return idx_sil_dict


def full_json_item_according_sil(idx_sil_dict, json_item):
    # 根据idx_sil_dict，在json_lst的对应下标处insert对应的ap或者sp
    for idx, sil in idx_sil_dict.items():
        # print(idx, sil)
        json_item["phs"].insert(int(idx), sil)
        json_item["is_slur"].insert(int(idx), 0)
        json_item["ph_dur"].insert(int(idx), 0)

        json_item["notes"].insert(int(idx), 0)
        json_item["notes_dur"].insert(int(idx), 0)
    # print(json_item)
    return json_item


def get_tg_json_align_meta(ori_meta_fn, tgt_meta_fn):
    # 如果目标文件存在，则删除
    if os.path.exists(tgt_meta_fn):
        os.remove(tgt_meta_fn)
    with open(ori_meta_fn, "r") as f:
        ori_datas = json.load(f)
    # cnt = 0
    for d in tqdm(ori_datas):
        # cnt += 1
        # if cnt>50:
        #     break
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
        tg = textgrid.TextGrid.fromFile(m4_tg_fn)
        word_iter = tg[0]
        phone_tier = tg[1]
        phs_in_tg = [itv.mark for itv in phone_tier]

        # whether tg consistent with json file
        # phs_in_tg should be included in phs
        only1, only2 = compare2re_lst(phs_in_tg, phs)
        if len(only1) != 0:
            # print(f'{phs_in_tg} in textgrid, but not in json file, which is not supposed to be')
            # print(len(phs_in_tg), phs_in_tg)
            # print(len(phs), phs)
            # print(is_slur)
            # print(only1)
            c_phs_in_json = phs
            c_phs_in_tg = phs_in_tg
            c_is_slur = is_slur
            c_item = d
            # try:
            new_tg_lst = get_new_tg(c_phs_in_json, c_phs_in_tg, c_is_slur, item_name)
            # except Exception as e:
            #     print(e, item_name)
            #     exit()

            idx_sil_dict = get_idx_sil(c_phs_in_json, new_tg_lst)
            d = full_json_item_according_sil(idx_sil_dict, c_item)
            assert (
                len(d["phs"])
                == len(d["is_slur"])
                == len(d["ph_dur"])
                == len(d["notes"])
                == len(d["notes_dur"])
            ), (
                d["item_name"],
                d["phs"],
                d["is_slur"],
                d["ph_dur"],
                d["notes"],
                d["notes_dur"],
            )
        else:
            pass
        with open(tgt_meta_fn, "a", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    root_m4_tg = "/Netdata/AudioData/m4singer"
    ori_meta_fn = "/Netdata/AudioData/m4singer/meta.json"
    align_meta_fn = "data/meta/m4-adjust-ori.json"
    get_tg_json_align_meta(ori_meta_fn, align_meta_fn)
