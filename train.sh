# stage1
python train.py --data_path ./data/mimic_report.json --unfreeze_ecg_model --batch_size 32 --accumulation_steps 8 --num_epochs 2 --output_dir /mnt/sda1/xxxx/output/anyECG/stage1
# stage2
ckpt_dir="/mnt/sda1/xxxx/output/anyECG/stage1/step_6000"
python train.py --data_path ./data/stage2.json --unfreeze_ecg_model --use_lora --projection_ckpt "${ckpt_dir}/projection.pth" --ecg_model_ckpt "${ckpt_dir}/ecg_model.pth" --batch_size 8 --accumulation_steps 8 --num_epochs 2 --output_dir /mnt/sda1/xxxx/output/anyECG/debug
# stage3
ckpt_dir="/mnt/sda1/xxxx/output/anyECG/stage2/step_28854"
python train.py --data_path ./data/stage3.json --unfreeze_ecg_model --use_lora --projection_ckpt "${ckpt_dir}/projection.pth" --ecg_model_ckpt "${ckpt_dir}/ecg_model.pth" --lora_ckpt $ckpt_dir --batch_size 8 --accumulation_steps 8 --num_epochs 1 --output_dir /mnt/sda1/xxxx/output/anyECG/stage3