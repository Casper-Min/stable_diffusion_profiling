# Profiling Example
# $ bash run_nsys_custom.sh name_prefix device_no num_of_prompts num_images_per_prompt num_inference_steps
# $ bash run_nsys_custom.sh test 0 1 1 50

name_prefix=$1
device_no=$2
batch_size=$3
num_images_per_prompt=$4
num_inference_steps=$5
CONFIG=${name_prefix}_prompt${batch_size}_iter${num_images_per_prompt}_step${num_inference_steps}
DIR=nsight_system_report/${CONFIG}
mkdir -p ${DIR}

nsys profile \
-o ${DIR}/${CONFIG} -f true \
-w true -t cuda,nvtx,cublas \
--capture-range=cudaProfilerApi --capture-range-end=stop-shutdown \
--nvtx-domain-include "nsys_prof" \
--gpu-metrics-device ${device_no} --gpu-metrics-frequency 10000 \
-s none --cpuctxsw none \
-e CUDA_VISIBLE_DEVICES=${device_no} \
python scripts/stable_diffusion_profile.py \
-o ${DIR} \
-p ${batch_size} \
-n ${num_images_per_prompt} \
-s ${num_inference_steps} \
-ci ${device_no}

# NVTX Range
# --capture-range=nvtx \
# --capture-range-end=stop-shutdown \
# --nvtx-capture="load_model@ncu_prof" \
# --nvtx-domain-include "nsys_prof" \
# --nvtx-domain-exclude "ncu_prof" \
# Capture Range
# -c cudaProfilerApi --capture-range-end stop-shutdown \
# NO CPU Backtrace
# -s none --cpuctxsw none \
# CUDA Backtrace
# --cudabacktrace all --cuda-memory-usage true \
# GPU Metric
# --gpu-metrics-device ${device_no} --gpu-metrics-frequency 10000 \
# NVTX Range
# --nvtx-capture range