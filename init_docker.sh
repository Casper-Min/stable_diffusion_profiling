cur_dir=$(pwd)
DIR=$cur_dir/host_dir
mkdir -p ${DIR}
docker build -t nsight:cuda12.4.1-ubt22.04-nsight12.4 .
docker run -it --privileged --gpus all --name sd_profile -v ${DIR}:/root/docker_dir nsight:cuda12.4.1-ubt22.04-nsight12.4 /bin/bash
