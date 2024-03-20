#!/bin/bash
###
 # @Author: hualizhou167 zhouHLzhou@163.com
 # @Date: 2024-03-20 16:48:06
 # @LastEditors: hualizhou167 zhouHLzhou@163.com
 # @LastEditTime: 2024-03-20 16:48:07
 # @FilePath: /BiSinger-archive/utils/get_pairs_db4.sh
 # @Description: 
 # 
 # Copyright (c) 2024 by Huali Zhou, All Rights Reserved.
### 

types=(CN EN MIX)
type=CN

if [ $type == CN ]
then
    sed -n '1,20000p' raw/中文女生DB-4/$type/ProsodyLabeling/text.txt | while read line1 && read line2; do
        # 用awk分离出句子号和文本内容
        id=$(echo $line1 | awk -F" " '{print $1}')
        text=$(echo $line1 | awk -F" " '{print $2}')
        pinyin=$(echo $line2 | cut -f2-)

        # 去掉文本中的韵律标注#1,#2,#3,#4
        text=$(echo "$text" | sed -E 's/#[1-4]//g')
        # 去除中文标点符号
        # 's/[”“：]//g'
        text=$(echo "$text" | sed -r 's/[，！。、——？）（]//g')
        text=$(echo "$text" | sed 's/\.\.\.//g; s/…//g; s/……//g; s/[”“：；]//g')
        # 把汉字之间添加空格
        spaced_text=$(echo "$text" | sed 's/./& /g' | sed 's/ $//')
        # 去除拼音中的声调标注1-6
        pinyin=$(echo $pinyin | sed 's/[1-6]//g')

        # 去除汉字和拼音最后的空格
        spaced_text=$(echo "$spaced_text" | sed 's/[[:space:]]*$//')
        pinyin=$(echo "$pinyin" | sed 's/[[:space:]]*$//')
        # 输出句子号和带空格的文本，格式为：000001|干 净 的 文 本|gan jing de wen ben
        printf "%s|%s|%s\n" "${id}" "${spaced_text}" "${pinyin}"
    done | awk '{print $0}' ORS='\n' > data/mfa_preparation/output_cn.txt
elif [ $type == EN ]
then
    # sed -n '1,10000p' raw/中文女生DB-4/$type/ProsodyLabeling/text.txt | while read line1 && read line2; do
    cat raw/中文女生DB-4/$type/ProsodyLabeling/text.txt | while read line1 && read line2; do
        # 用awk分离出句子号和文本内容
        id=$(echo $line1 | awk '{print $1}')
        text=$(echo $line1 |  awk '{print substr($0, length($1)+2)}')
        phone=$(echo $line2 | cut -f2-)

        # 去掉文本内容中的韵律标注符号，单词“空格”、韵律短语(/)、语调短语(%)、连读(-)
        text=$(echo "$text" | sed -E 's/[-%\/.~![:punct:]]//g')

        # 去掉音标中：单词之间以“/”分开，音素之间以空格分开,音节以"."分隔。
        # 去掉英文重音：0代表非重音，1代表重音，2代表次重音，3代表句重音。
        phone=$(echo "$phone" | sed -E 's/[0123\/.]//g')

        # 去除汉字和拼音最后的空格
        text=$(echo "$text" | sed 's/[[:space:]]*$//')
        phone=$(echo "$phone" | sed 's/[[:space:]]*$//')
        # 输出句子号和带空格的文本，格式为：300004|his sister Sara asked|HH IH Z S IH S T ER S EH R AH AE S K T
        printf "%s|%s|%s\n" "${id}" "${text}" "${phone}"
    done | awk '{print $0}' ORS='\n' > data/mfa_preparation/output_en.txt
fi
