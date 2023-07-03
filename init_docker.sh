cur_dir=$(pwd)
DIR=$cur_dir/host_dir
mkdir -p ${DIR}
docker build -t nsight:cuda12.1.1-ubt22.04-nsight12.1 .
docker run -it --privileged --gpus all --name sd_profile -v ${DIR}:/root nsight:cuda12.1.1-ubt22.04-nsight12.1 /bin/bash