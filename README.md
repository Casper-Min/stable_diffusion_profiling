# Stable Diffusion Docker/Profiling Setting

Based on the repository [`transformers`](https://github.com/huggingface/transformers) [`diffusers`](https://github.com/huggingface/diffusers) [`NVTX`](https://github.com/NVIDIA/NVTX)

### Modification to diffuesrs
1. Code for NVTX manual-annotation is added.
2. Baseline(unoptimized) attention processor "AttnProcessor" is used for attention block.

### Modification to NVTX
1. Domain of NVTX annotation from push_range/pop_range is set as "nsys_prof".
2. List of file/function described below is ignored in NVTX auto-annotation.

   File : "module.py","nvtx.py","_contextlib.py","cached.py","grad_mode.py","threading.py","_monitor.py"
   
   Function : "decorate_context","_call_impl"

## Docker Setting

Modifiy docker volume directory in init.sh, then

```
bash init_docker.sh
```

## Profiling Setting

After attach container, activate conda environment "sd_profile", then

```
cd /home/workspace/
bash init_profiling.sh
```

## Installation Test

Check whether docker container/conda environment is properly installed.

```
cd /home/workspace/
python test.py
```

# Profiling Example

Stable diffusion scripts are located in /home/workspace/profiling/scripts

Profiling scripts are located in /home/workspace/profiling/ 

## Nsight System example
1. run_nsys_nvtx_range.sh : Code with NVTX range domain "nsys_prof" will be profiled.
2. run_nsys_nvtx_all_range.sh : Entire code will be profiled.
3. run_nsys_nvtx_range_gpu_metric.sh : Code with NVTX range domain "nsys_prof" will be profiled + GPU metric will be collected.
4. run_nsys_nvtx_range_sweep.sh : For design space exploration.
5. run_nsys_custom.sh : stable diffusion with custome pipeline will be profiled.
```
mv /home/workspace/profiling/
bash run_nsys_custom.sh name_prefix device_no num_of_prompts num_images_per_prompt num_inference_steps
bash run_nsys_custom.sh test 0 1 1 50
```

## Nsight Compute example
As profiling entire script with Nsight Compute needs more than 2 days, you should profiling on a limited range of NVTX annotation.

run_ncu_nvtx_range.sh : Code with NVTX range domain "ncu_prof" and name "denoiser_step_2" will be profiled.

```
mv /home/workspace/profiling/
bash run_ncu_nvtx_range.sh name_prefix device_no num_of_prompts num_images_per_prompt num_inference_steps replay
bash run_ncu_nvtx_range.sh test 0 1 1 50 kernel full
```
