import json
import os
from tqdm import tqdm

json_files = [
    "data/meta/db4cn-wdb.json",
    "data/meta/db4en-wdb.json",
    "data/meta/db4cn-shift-wdb.json",
    "data/meta/db4en-shift-wdb.json",
    "data/meta/m4-proportional.json",
]

# 创建一个空的json列表用于存放所有json文件的内容
merged_data = []

# 遍历所有的json文件，将其内容读取并添加到merged_json列表中
for file in json_files:
    with open(file, "r", encoding="utf-8") as f:
        file_content = f.readlines()
        merged_data.extend(file_content)

# 添加speechsing标志，并写入目标文件
tgt_fn = "target_combination.json"
# 如果文件存在，则删除文件
if os.path.exists(tgt_fn):
    os.remove(tgt_fn)

for line in tqdm(merged_data):
    item = eval(line)
    singer, song_name, sent_id = item["item_name"].split("#")
    if singer == "db4":
        if song_name.endswith("shift"):
            item["speechsing"] = 2  # 2表示伪歌声
        else:
            item["speechsing"] = 0  # 0表示speech
    else:
        item["speechsing"] = 1  # 1表示singing
    with open(tgt_fn, "a", encoding="utf-8") as f:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

# 将合并后的json列表写入新的json文件
# new_json_fn = "m4_db4_ori_shift_wdb_ss.json"
# with open(new_json_fn, 'w', encoding='utf-8') as f:
#     for info in tqdm(merged_data):
#         f.write(info)
