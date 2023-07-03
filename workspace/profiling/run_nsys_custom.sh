# Profiling Example
# $ bash run_nsys_custom.sh name_prefix device_no num_of_prompts num_images_per_prompt num_inference_steps
# $ bash run_nsys_custom.sh test 0 1 1 50

name_prefix=$1
device_no=$2
batch_size=$3
num_images_per_prompt=$4
num_inference_steps=$5
CONFIG=custom_${device_name}_batch${batch_size}_iter${num_images_per_prompt}_step${num_inference_steps}
DIR=nsight_system_report/${CONFIG}
mkdir -p ${DIR}

nsys profile \
-o ${DIR}/${CONFIG} -f true \
-w true -t cuda,nvtx,cublas \
-c cudaProfilerApi --capture-range-end stop \
-s none --cpuctxsw none \
-e CUDA_VISIBLE_DEVICES=${device_no} \
python scripts/code_review_custom_pipeline_profiling.py \
-o ${DIR} \
-p ${batch_size} \
-n ${num_images_per_prompt} \
-s ${num_inference_steps} \
-ci ${device_no}
# --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown \
# --nvtx-domain-include "nsys_prof" \