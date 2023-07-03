conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y accelerate scipy protobuf datasets Cython -c conda-forge 
cd libs && git clone https://github.com/huggingface/transformers.git
cd transformers && pip install -e . && cd ..
cd diffusers && pip install -e . && cd ..
cd NVTX/python && python setup.py install && cd ../../../
python test.py