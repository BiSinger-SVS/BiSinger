#!/bin/bash
types=(CN EN MIX)
type=CN

if [ $type == CN ]
then
    wav_dir=/NASdata/AudioData/mandarin/speech_synthesis/中文女生DB-4/cn_44k
    while read -r line; do
        id=$(echo "$line" | cut -d '|' -f 1)
        text=$(echo "$line" | cut -d '|' -f 3)
        echo $text > ../mfa/db4-cn/input/$id.txt
        cp $wav_dir/$id.wav ../mfa/db4-cn/input/$id.wav
    done < data/mfa_preparation/output_cn.txt
elif [ $type == EN ]
then
    wav_dir=/NASdata/AudioData/mandarin/speech_synthesis/中文女生DB-4/en_44k
    while read -r line; do
        id=$(echo "$line" | cut -d '|' -f 1)
        text=$(echo "$line" | cut -d '|' -f 2)
        echo $text > ../mfa/db4-en/input/$id.txt
        cp $wav_dir/$id.wav ../mfa/db4-en/input/$id.wav
    done < data/mfa_preparation/output_en.txt
fi
