#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python run_speech_recognition_rnnt.py \
        --config_path="conf/conformer_transducer_bpe_xlarge.yaml" \
        --model_name_or_path="stt_en_conformer_transducer_xlarge" \
        --dataset_name="mozilla-foundation/common_voice_9_0" \
        --tokenizer_path="tokenizer" \
        --vocab_size="1024" \
        --num_train_epochs="4" \
        --evaluation_strategy="epoch" \
        --dataset_config_name="en" \
        --train_split_name="train" \
        --eval_split_name="validation" \
        --test_split_name="test" \
        --text_column_name="sentence" \
        --output_dir="./" \
        --run_name="rnnt-cv9-baseline" \
        --wandb_project="rnnt" \
        --per_device_train_batch_size="8" \
        --per_device_eval_batch_size="4" \
        --logging_steps="25" \
        --learning_rate="1e-4" \
        --warmup_steps="2000" \
        --save_steps="200000" \
        --evaluation_strategy="steps" \
        --eval_steps="80000" \
        --report_to="wandb" \
        --preprocessing_num_workers="4" \
        --fused_batch_size="8" \
        --fuse_loss_wer \
        --group_by_length \
        --overwrite_output_dir \
        --fp16 \
        --do_lower_case \
        --do_train \
        --do_eval \
        --do_predict \
        --push_to_hub \
        --use_auth_token
