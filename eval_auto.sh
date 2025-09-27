export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
# model=qwen
model=internvl
# model=glm
image_root_dir=/root/siton-data-51d3ce9aba3246f88f64ea65f79d5133/images


label=grid_consistency_subclip_extended_73

for epoch in {40..320..40}; do
    python eval.py \
        --image_dir ${image_root_dir}/${label}/${epoch} \
        --result_file eval_res/${model}/${label}/${epoch}.jsonl
done