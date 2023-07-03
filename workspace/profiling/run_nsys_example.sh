# Nsight Conpute Profiling Example
# Check Nsight Systems document for further information
# https://docs.nvidia.com/nsight-systems/UserGuide/index.html
# Check Nsight Systems profiling commands for Pytorch scripts
# https://gist.github.com/mcarilli/376821aa1a7182dfcf59928a7cde3223

# Basic Example
nsys profile nsys-args... python script.py script-args...

# Nsight Systems Args
-o ${DIR}/${CONFIG} # output file
-f true # overwrite existing output file
-w true # Don't suppress app's console output.

# Trace
-t cuda,nvtx,cublas

# Do Not Backtrace CPU
-s none
--cpuctxsw none

# Backtrace CUDA API
-s cpu # Sample the cpu stack periodically
--cudabacktrace all
--cuda-memory-usage true

# Set Capture Range
--capture-range=cudaProfilerApi # Only start profiling when the app calls cudaProfilerStart...
--capture-range-end=stop-shutdown # ...and end profiling when the app calls cudaProfilerStop.

--capture-range=nvtx
--nvtx-capture="range@domain"

# Set NVTX Domain
--nvtx-domain-include "nsys_prof"
--nvtx-domain-exclude "ncu_prof"

# Record GPU Metric
--gpu-metrics-device ${device_no}
--gpu-metrics-frequency 10000

# Application Environment Variable
-e CUDA_VISIBLE_DEVICES=${device_no}