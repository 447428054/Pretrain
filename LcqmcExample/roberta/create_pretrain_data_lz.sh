#!/usr/bin/env bash

python3 create_pretraining_data.py --do_whole_word_mask=True --input_file=./raw_text/lcqmc4pretrain.txt \
--output_file=./tf_records_all/tf_lcqmc.tfrecord --vocab_file=./resources/vocab.txt \
--do_lower_case=True --max_seq_length=256 --max_predictions_per_seq=23 --masked_lm_prob=0.10  --random_seed=12345  --dupe_factor=5
