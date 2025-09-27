# export CUDA_VISIBLE_DEVICES=0

# model=black-forest-labs/FLUX.1-dev
model=/root/siton-data-51d3ce9aba3246f88f64ea65f79d5133/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21

label=grid_times_consistency_plus_subclip_half
output_dir=/root/siton-data-51d3ce9aba3246f88f64ea65f79d5133/images
checkpoint_dir=/root/siton-data-51d3ce9aba3246f88f64ea65f79d5133/checkpoints/Flow_GRPO/grid-consistency-subclip/flux-7gpu-2by2-half_grid-times-consistency-plus-clipT/checkpoints

for epoch in {80..240..40} ; do
    python image_gen.py \
        --model ${model} \
        --output_dir ${output_dir}/${label}_${epoch} \
        --lora_path ${checkpoint_dir}/checkpoint-${epoch}/lora/
done