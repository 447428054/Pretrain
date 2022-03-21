export BERT_BASE_DIR=/home/root1/lizheng/pretrainModels/tensorflow/chinese/roberta_zh_l12
python3 run_pretraining.py --input_file=./tf_records_all/tf_lcqmc.tfrecord  \
--output_dir=./save --do_train=True --do_eval=True --bert_config_file=$BERT_BASE_DIR/bert_config.json \
--train_batch_size=2 --max_seq_length=256 --max_predictions_per_seq=23 \
--num_train_steps=200000 --num_warmup_steps=10000 --learning_rate=1e-4    \
--save_checkpoints_steps=3000  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt