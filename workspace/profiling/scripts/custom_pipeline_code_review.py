import argparse
from sympy import false
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers.utils import randn_tensor
from tqdm.auto import tqdm
from PIL import Image

import torch.cuda.nvtx as torch_nvtx
import nvtx as cuda_nvtx
# from cuda import cuda, cudart, nvrtc
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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
    parser.add_argument(
        "-pr",
        "--pre_defined_pipeline",
        type=bool,
        default=False,
        help="Use pre-defined_pipeleine for custom_pipeline.",
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

@torch.no_grad()
def main():
    
    # 1. Configure stable diffusion paramters
    args = parse_args()
    
    model_id = args.model_id                            # model id of stable diffusion model #
    height = args.height                                # default height of Stable Diffusion #
    width = args.width                                  # default width of Stable Diffusion #
    prompt = [args.prompt]                              # The prompt or prompts to guide the image generation #
    num_images_per_prompt = args.images_num             # Number of images to generate per prompt #
    num_inference_steps = args.steps-1                  # Number of denoising steps #
    guidance_scale = args.guidance                      # Scale for classifier-free guidance #
    generator = torch.manual_seed(args.seed)            # Seed generator to create the inital latent noise #
    torch_device = torch.device("cuda", args.cuda_id)   # GPU device #
    batch_size = len(prompt)                            # Number of prompts to generate images #
    with_predefined_pipeline = False                    # Use pre-defined pipeline or not #
    output_path = args.path                             # Path to save the generated images #
    
    # 2. Construct stable diffusion pipeline
    # pipeline: text_encoder&tokenizer + unet&scheduler + vae
    # use pre-defined diffusion pipeline 
    if with_predefined_pipeline == True:
        pipeline = StableDiffusionPipeline.from_pretrained(model_id)
    
    # OR construct custom stable diffusion pipeline
    elif with_predefined_pipeline == False:
        # 2-1. Load the tokenizer and text encoder to tokenize and encode the text.
        # text_encoder: Other diffusion models may use other encoders such as BERT(Default : CLIP)
        # tokenizer: It must match the one used by the text_encoder model(Default : CLIPtokenizer)
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

        # 2-2. The UNet model for generating the latents.
        # unet: Model used to generate the latent representation of the input(Default : UNET))
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        
        # 2-3. The scheduler for denoising latent vector.
        # scheduler: Scheduling algorithm used to progressively add noise to the image during training
        # (Deault : PNDM)
        scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        # 2-4. Load the autoencoder model which will be used to decode the latents into image space. 
        # vae: Autoencoder module that we'll use to decode latent representations into real images.
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        height = args.height or unet.config.sample_size * vae_scale_factor
        width = args.width or unet.config.sample_size * vae_scale_factor
    
    # Start of profiling
    # cudart.cudaProfilerStart()
    torch.cuda.profiler.cudart().cudaProfilerStart()
    pr = cuda_nvtx.Profile()
    pr.enable()  # start annotating function calls
    
    # 3. Load models to GPU
    # 3-1 Load pre-defined diffusion pipeline
    if with_predefined_pipeline == True:
        cuda_nvtx.push_range(message="pipeline_model_load", color="blue")
        pipeline.to(torch_device)
        cuda_nvtx.pop_range()
        
    # 3-2 OR Load custom diffusion pipeline
    elif with_predefined_pipeline == False:
        cuda_nvtx.push_range(message="model_load", color="cyan")
        
        cuda_nvtx.push_range(message="CLIP_model_load", color="blue")
        text_encoder.to(torch_device)
        cuda_nvtx.pop_range()
        
        cuda_nvtx.push_range(message="UNET_model_load", color="red")
        unet.to(torch_device)
        cuda_nvtx.pop_range()
        
        cuda_nvtx.push_range(message="VAE_model_load", color="green")
        vae.to(torch_device)
        cuda_nvtx.pop_range()
        
        cuda_nvtx.pop_range()

    if with_predefined_pipeline == True:
        image = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
            ).images
        
        # pr.disable()  # stop annotating function calls
        # cudart.cudaProfilerStop()
        torch.cuda.profiler.cudart().cudaProfilerStop()
        grid = image_grid(image, rows=batch_size, cols=num_images_per_prompt)
        grid.save(output_path+f"/predefined_result_step_{num_inference_steps}.png")
    
    elif with_predefined_pipeline == False:
        # 4. Tokenize the text and generate the embeddings from the prompt:
        # 4-1. text to token
        
        cuda_nvtx.push_range(message="encoder", color="red")
        
        cuda_nvtx.push_range(message="text_to_token", color="blue")
        text_input = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_input_ids = text_input.input_ids
        cuda_nvtx.pop_range()
        
        # 4-2. token to embedding
        cuda_nvtx.push_range(message="token_to_embedding", color="red")
        # with torch.no_grad():
        text_embeddings = text_encoder(text_input_ids.to(torch_device))[0]
        # text_embeddings = text_embeddings
        cuda_nvtx.pop_range()
        
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        cuda_nvtx.push_range(message="embedding_postprocess", color="green")
        text_embeddings = text_embeddings.to(dtype=text_encoder.dtype, device=torch_device)
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
        cuda_nvtx.pop_range()
        
        # get unconditional embeddings for classifier free guidance
        # 4-3. text to token
        cuda_nvtx.push_range(message="text_to_token", color="blue")
        uncond_tokens = [""] * batch_size
        max_length = text_embeddings.shape[1]
        uncond_input = tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        uncond_input_ids = uncond_input.input_ids
        cuda_nvtx.pop_range()
        
        # 4-4. token to embedding
        cuda_nvtx.push_range(message="token_to_embedding", color="red")
        # with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input_ids.to(torch_device))[0]
        # uncond_embeddings = uncond_embeddings
        cuda_nvtx.pop_range()
        
        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        cuda_nvtx.push_range(message="embedding_postprocess", color="green")
        seq_len = uncond_embeddings.shape[1]
        uncond_embeddings = uncond_embeddings.to(dtype=text_encoder.dtype, device=torch_device)
        uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
        uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)
        
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        cuda_nvtx.pop_range()
        cuda_nvtx.pop_range()
        
        # 5. Initialize denoising network
        # 5-1. Generate random noise
        cuda_nvtx.push_range(message="initialize denoising network", color="purple")
        import pdb; pdb.set_trace()
        cuda_nvtx.push_range(message="creat_random_noise", color="blue")
        shape = (batch_size * num_images_per_prompt, unet.config.in_channels, height // vae_scale_factor, width // vae_scale_factor)
        latents = randn_tensor(
            shape,
            generator=generator,
            device=torch_device,
            dtype=text_embeddings.dtype
        )
        cuda_nvtx.pop_range()
        # 5-2. scale the initial noise by the standard deviation required by the scheduler
        cuda_nvtx.push_range(message="init_latent_noise", color="red")
        latents = latents * scheduler.init_noise_sigma
        cuda_nvtx.pop_range()
        # 5-3. initialize the scheduler with our chosen num_inference_steps
        cuda_nvtx.push_range(message="init_scheduler", color="green")
        scheduler.set_timesteps(num_inference_steps)
        cuda_nvtx.pop_range()
        cuda_nvtx.pop_range()
        
        # 6. denoising network loop(for N steps)
        cuda_nvtx.push_range(message=f"denoise_loop", color="blue")
        ii = 0
        for t in tqdm(scheduler.timesteps):
            cuda_nvtx.push_range(message=f"loop_{ii}", color="cyan")
            # 6-1. expand the latents if we are doing classifier-free guidance
            # to avoid doing two forward passes.            
            cuda_nvtx.push_range(message=f"preprocess_latent_{t}", color="orange")
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
            cuda_nvtx.pop_range()
            # 6-2. UNET : predict the noise residual
            # with torch.no_grad():
            cuda_nvtx.push_range(message=f"unet_{ii}", color="green")
            noise_pred = unet(
                latent_model_input,
                timestep=t,
                encoder_hidden_states=text_embeddings,
                return_dict=False
            )[0]
            cuda_nvtx.pop_range()
            
            # 6-3.reflect guidance scale on predicted noise to perform classifier-free guidance
            cuda_nvtx.push_range(message=f"guidance_{ii}", color="yellow")
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            cuda_nvtx.pop_range()
            
            # 6-4. subtract sampe x_(t) with predicted noise to generate sameple x_(t-1)
            cuda_nvtx.push_range(message=f"compute_denoise_{ii}", color="darkgreen")
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            cuda_nvtx.pop_range()
            
            cuda_nvtx.pop_range()
            ii += 1
            
        cuda_nvtx.pop_range()
        # cuda_nvtx.end_range(denoise_loop)
        
        # 7. Decode the image 
        # 7-1. scale the denoised latent by scaling factor required by the VAE
        
        cuda_nvtx.push_range(message="decoder", color="red")
        
        cuda_nvtx.push_range(message="scale_latent", color="blue")
        latents = 1 / vae.config.scaling_factor * latents
        cuda_nvtx.pop_range()
        # 7-2. decode the image latents with vae
        cuda_nvtx.push_range(message="decode_latent", color="green")
        image = vae.decode(latents, return_dict=False)[0]
        cuda_nvtx.pop_range()
        
        cuda_nvtx.pop_range()
        
        # 8. Post-process the image    
        # convert the image to PIL and save it
        cuda_nvtx.push_range(message="post-process", color="purple")
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        cuda_nvtx.pop_range()
        
        pr.disable()  # stop annotating function calls
        # cudart.cudaProfilerStop()
        torch.cuda.profiler.cudart().cudaProfilerStop()
        
        grid = image_grid(pil_images, rows=1, cols=num_images_per_prompt)
        grid.save(output_path+f"/custom_result_step_{num_inference_steps}.png")
    
    return grid
    

if __name__ == "__main__":
    main()