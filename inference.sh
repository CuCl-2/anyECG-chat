export CUDA_VISIBLE_DEVICES=3
model_name="anyECG-chat"
# model_name="pulse"
ckpt_dir="/mnt/sda1/xxxx/output/anyECG/stage3/step_17057"
text_embedding_model_dir="/mnt/sda1/xxxx/huggingface"
for temperature in 0.6
do
    for text_embedding_model in BioBERT-mnli-snli-scinli-scitail-mednli-stsb
    do
        for dataset in cpsc csn ptbxl european_st_t mit_bih_st mit_bih_arrhythmia european_st_t_long mit_bih_st_long mit_bih_arrhythmia_long ecgqa mimic-multi
        do
            if [ "$dataset" = "ptbxl" ]; then
                for subtype in  super-diag sub-diag form rhythm
                do
                    python inference.py --dataset $dataset --dataset_subtype $subtype --model_name $model_name --projection_ckpt "${ckpt_dir}/projection.pth" --ecg_model_ckpt "${ckpt_dir}/ecg_model.pth" --lora_ckpt $ckpt_dir --temperature $temperature --text_embedding_model "${text_embedding_model_dir}/${text_embedding_model}" \
                    --result_path "${ckpt_dir}/${dataset}_${subtype}_${text_embedding_model}_tem${temperature}.json" 
                    # --result_path "/mnt/sda1/xxxx/output/anyECG/other_baseline/${model_name}/${dataset}_${subtype}_${text_embedding_model}_tem${temperature}.json" --eval_batch_size 10
                done
            else
                python inference.py --dataset $dataset --dataset_subtype all --model_name $model_name --projection_ckpt "${ckpt_dir}/projection.pth" --ecg_model_ckpt "${ckpt_dir}/ecg_model.pth" --lora_ckpt $ckpt_dir --temperature $temperature --text_embedding_model "${text_embedding_model_dir}/${text_embedding_model}" \
                --result_path "${ckpt_dir}/${dataset}_${text_embedding_model}_tem${temperature}.json" 
                # --result_path "/mnt/sda1/xxxx/output/anyECG/other_baseline/${model_name}/${dataset}_${text_embedding_model}_tem${temperature}.json" --eval_batch_size 10
            fi
        done
    done
done

# export CUDA_VISIBLE_DEVICES=2
# ckpt_dir="/mnt/sda1/xxxx/output/anyECG/stage3/step_17057"
# for dataset in mit_bih_st  #  european_st_t_long mit_bih_st_long mit_bih_arrhythmia_long
# do
#     python inference.py --dataset $dataset --dataset_subtype all --projection_ckpt "${ckpt_dir}/projection.pth" --ecg_model_ckpt "${ckpt_dir}/ecg_model.pth" --lora_ckpt $ckpt_dir --temperature 0.6  --result_path "${ckpt_dir}/${dataset}_mask_random_tem.json" --mask_random_non_zero_lead &
#     python inference.py --dataset $dataset --dataset_subtype all --projection_ckpt "${ckpt_dir}/projection.pth" --ecg_model_ckpt "${ckpt_dir}/ecg_model.pth" --lora_ckpt $ckpt_dir --temperature 0.6  --result_path "${ckpt_dir}/${dataset}_mask_first_tem.json" --mask_first_non_zero_lead &
#     python inference.py --dataset $dataset --dataset_subtype all --projection_ckpt "${ckpt_dir}/projection.pth" --ecg_model_ckpt "${ckpt_dir}/ecg_model.pth" --lora_ckpt $ckpt_dir --temperature 0.6  --result_path "${ckpt_dir}/${dataset}_mask_second_tem.json" --mask_second_non_zero_lead &
# done
# wait