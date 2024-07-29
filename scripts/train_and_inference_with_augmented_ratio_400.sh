accelerate launch STEP_3_train_SLU_model.py \
  --model_name_or_path "openai/whisper-base.en" \
  --train_data "./data/train_data_augmented_4.json" \
  --eval_data "./data/eval_data.json" \
  --eval_steps 5000 \
  --save_steps 5000 \
  --warmup_steps 5000 \
  --learning_rate 0.00003 \
  --logging_steps 100 \
  --save_total_limit 1 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --dataloader_num_workers 2 \
  --gradient_accumulation_steps 1 \
  --ddp_timeout 7200 \
  --dtype "float16" \
  --attn_implementation "sdpa" \
  --output_dir "./outputs/SLU_model_org_data_with_augmented_ratio_400_epochs_2" \
  --do_train \
  --do_eval \
  --gradient_checkpointing \
  --overwrite_output_dir \
  --report_to "none" \
  --seed 0

python utils/remove_suppress_tokens.py --base_folder ./outputs/SLU_model_org_data_with_augmented_ratio_400_epochs_2

python inference.py --model_path=outputs/SLU_model_org_data_with_augmented_ratio_400_epochs_2 --test_data data/test_data.json 

python evaluate_EM_metric_newline.py --prediction outputs/SLU_model_org_data_with_augmented_ratio_400_epochs_2/predictions.json --is_full_form --pred_is_dict --gold_is_dict > outputs/SLU_model_org_data_with_augmented_ratio_400_epochs_2/test_results.txt
