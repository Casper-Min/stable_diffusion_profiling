import torch
import torch.cuda.nvtx as torch_nvtx
import nvtx as cuda_nvtx
from diffusers import DiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"
prompt = "a photo of an astronaut riding a horse on mars"

pipe = DiffusionPipeline.from_pretrained(model_id)

torch.cuda.profiler.cudart().cudaProfilerStart()
pr = cuda_nvtx.Profile()
pr.enable()  # start annotating function calls

pipe.to("cuda")
image = pipe(prompt).images[0]

pr.disable()  # stop annotating function calls

image.save(f"astronaut_rides_horse.png")