# Profiling Example
# $ bash run_ncu_nvtx_range.sh name_prefix device_no num_of_prompts num_images_per_prompt num_inference_steps replay
# $ bash run_ncu_nvtx_range.sh test 0 1 1 50 kernel full

name_prefix=$1
device_no=$2
batch_size=$3
num_images_per_prompt=$4
num_inference_steps=$5
replay=$6
set=$7
CONFIG=${name_prefix}_prompt${batch_size}_iter${num_images_per_prompt}_step${num_inference_steps}_replay_${replay}_set_${set}
DIR=nsight_compute_report/${CONFIG}
mkdir -p ${DIR}

CUDA_VISIBLE_DEVICES=${device_no} ncu \
-o ${DIR}/${CONFIG} -f --target-processes all \
--replay-mode ${replay} --set ${set} \
--nvtx --nvtx-include "ncu_prof@denoiser_step_2" \
python scripts/stable_diffusion_profile.py \
-o ${DIR} \
-p ${batch_size} \
-n ${num_images_per_prompt} \
-s ${num_inference_steps} \
-ci ${device_no}

# Test Range
# --nvtx --nvtx-include "ncu_prof@denoiser_step_2" \
# --nvtx --nvtx-include "ncu_prof@denoiser_step_2,unet_block_1" \
# --nvtx --nvtx-include "ncu_prof@denoiser_step_2,attn_block_2" \
# --nvtx --nvtx-include "ncu_prof@denoiser_step_2,unet_block_1,attn_block_2" \
