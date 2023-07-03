import argparse
import os
import torch
from PIL import Image
from diffusers import StableDiffusionPipelineProfile
import nvtx as cuda_nvtx

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="a photograph of an astronaut riding a horse",
        help="Text used to generate images.",
    )
    parser.add_argument(
        "-n",
        "--images_num",
        type=int,
        default=1,
        help="How much images to generate.",
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=50,
        help="The number of denoising steps.",
    )
    parser.add_argument(
        "-wi",
        "--width",
        type=int,
        default=512,
        help="The width in pixels of the generated image.",
    )
    parser.add_argument(
        "-he",
        "--height",
        type=int,
        default=512,
        help="The height in pixels of the generated image.",
    )
    parser.add_argument(
        "-g",
        "--guidance",
        type=float,
        default=7.5,
        help="Higher guidance scale encourages to generate images that are closely linked to the text.",
    )
    parser.add_argument(
        "-sd",
        "--seed",
        type=int,
        default=42,
        help="Seed for random process.",
    )
    parser.add_argument(
        "-ci",
        "--cuda_id",
        type=int,
        default=0,
        help="cuda_id.",
    )

    parser.add_argument(
        "-o",
        "--path",
        type=str,
        default="outputs",
        help="output path",
    )
    args = parser.parse_args()
    return args

def image_grid(imgs, rows, cols):
    if not len(imgs) == rows * cols:
        raise ValueError("The specified number of rows and columns are not correct.")

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def createDirectory(directory): 
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def find_next_available_folder(base_name): # 이미 있는 경우 다른 숫자 붙여서 경로 생성
    folder_name = base_name
    count = 1
    while os.path.exists(folder_name):
        folder_name = f"{base_name}{count}"
        count += 1
    return folder_name

def main():
    
    args = parse_args()
    
    model_id = args.model_id                            
    prompt = [args.prompt]                            
    height = args.height                         
    width = args.width                               
    num_inference_steps = args.steps                    
    guidance_scale = args.guidance                    
    num_images_per_prompt = args.images_num         
    batch_size = len(prompt)                           
    torch_device = torch.device("cuda", args.cuda_id)  
    generator = torch.manual_seed(args.seed)         
    output_path = args.path 
        
    pipeline = StableDiffusionPipelineProfile.from_pretrained(model_id)
    pipeline.to(torch_device)

    image = pipeline(
        prompt=prompt,
        height=height,
        width=width,
        num_images_per_prompt=num_images_per_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
        ).images
    grid = image_grid(image, rows=batch_size, cols=num_images_per_prompt)
    grid.save(output_path+f"/result_step_{num_inference_steps}.png")

if __name__ == "__main__":
    main()
    


    
    # batch_size = len(prompt)                            # Number of prompts to generate images #
    # torch_device = torch.device("cuda", args.cuda_id)   # GPU device #
    # output_path = args.path                             # Path to save the generated images #
    
    # model_id = args.model_id                            # model id of stable diffusion model #
    # height = args.height                                # default height of Stable Diffusion #
    # width = args.width                                  # default width of Stable Diffusion #
    # prompt = [args.prompt]                              # The prompt or prompts to guide the image generation #
    # num_images_per_prompt = args.images_num             # Number of images to generate per prompt #
    # num_inference_steps = args.steps                    # Number of denoising steps #
    # guidance_scale = args.guidance                      # Scale for classifier-free guidance #
    # generator = torch.manual_seed(args.seed)            # Seed generator to create the inital latent noise #
    