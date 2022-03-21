export CUDA_VISIBLE_DEVICES=1
export BERT_BASE_DIR=/home/root1/lizheng/pretrainModels/tensorflow/chinese/albert_base_zh_additional_36k_steps
python3 run_pretraining_google.py --input_file=./data/tf_lcqmc.tfrecord --eval_batch_size=64 \
--output_dir=./savealbert_base --do_train=True --do_eval=True --albert_config_file=$BERT_BASE_DIR/albert_config_base.json  --export_dir=./export_albert \
--train_batch_size=2 --max_seq_length=512 --max_predictions_per_seq=20 \
--num_train_steps=125000 --num_warmup_steps=12500 --learning_rate=0.00176   \
--save_checkpoints_steps=2000 --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt --max_predictions_per_seq 51