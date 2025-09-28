export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
# model=qwen2_5
# model=internvl3
model=internvl3_5
# model=glm
image_root_dir=/root/siton-data-51d3ce9aba3246f88f64ea65f79d5133/images

python eval.py \
    --image_dir ${image_root_dir}/base/0 \
    --result_file eval_res/${model}/base/0.jsonl

label=grid_times_consistency_plus_subclip_half

for epoch in {40..240..40}; do
    python eval.py \
        --image_dir ${image_root_dir}/${label}/${epoch} \
        --result_file eval_res/${model}/${label}/${epoch}.jsonl
done


label=grid_consistency_subclip_half_flexible_leq4

for epoch in {80..220..20}; do
    python eval.py \
        --image_dir ${image_root_dir}/${label}/${epoch} \
        --result_file eval_res/${model}/${label}/${epoch}.jsonl
done


label=grid_consistency_subclip_half

for epoch in 60 80 120; do
    python eval.py \
        --image_dir ${image_root_dir}/${label}/${epoch} \
        --result_file eval_res/${model}/${label}/${epoch}.jsonl
done

label=grid_consistency_subclip_extended_73

for epoch in {40..320..40}; do
    python eval.py \
        --image_dir ${image_root_dir}/${label}/${epoch} \
        --result_file eval_res/${model}/${label}/${epoch}.jsonl
done

label=consistency_subclipT_half

for epoch in {81..121..20}; do
    python eval.py \
        --image_dir ${image_root_dir}/${label}/${epoch} \
        --result_file eval_res/${model}/${label}/${epoch}.jsonl
done