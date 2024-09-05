import os
import time
import torch
from datetime import datetime
from diffusers import (
    StableDiffusionXLPipeline, 
    LMSDiscreteScheduler, DPMSolverMultistepScheduler, DDIMScheduler,
    EulerDiscreteScheduler, PNDMScheduler, EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler, DEISMultistepScheduler, DPMSolverSinglestepScheduler,
    KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler,
    UniPCMultistepScheduler, AutoencoderKL
)

# Load the VAE and ProteusV0.4 model
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
model_id = "dataautogpt3/ProteusV0.4"
pipe = StableDiffusionXLPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Disable safety checker
pipe.safety_checker = None

# Define the prompt
prompt = "A futuristic cityscape with tall skyscrapers, flying cars, and a bustling market with diverse characters."
negative_prompt = "nsfw, bad quality, bad anatomy, worst quality, low quality, low resolutions, extra fingers, blur, blurry, ugly, wrongs proportions, watermark, image artifacts, lowres, ugly, jpeg artifacts, deformed, noisy image"

# Seed for reproducibility
seed = 123456
torch.manual_seed(seed)  # Set the global seed for reproducibility
generator = torch.Generator(device="cuda").manual_seed(seed)  # Create a reproducible generator

# Define an array of schedulers with their corresponding scheduler classes
schedulers = [
    ("LMS", LMSDiscreteScheduler),
    ("DPM++ 2M Karras", DPMSolverMultistepScheduler),
    ("DDIM", DDIMScheduler),
    ("Euler a", EulerAncestralDiscreteScheduler),
    ("Euler", EulerDiscreteScheduler),
    ("Heun", HeunDiscreteScheduler),
    ("DEIS", DEISMultistepScheduler),
    ("DPM Solver++", DPMSolverSinglestepScheduler),
    ("KDPM2", KDPM2DiscreteScheduler),
    ("KDPM2 Ancestral", KDPM2AncestralDiscreteScheduler),
    ("UniPC", UniPCMultistepScheduler),
    ("PNDM", PNDMScheduler)
]

# Ensure schedulers directory exists
output_dir = "./text2image_scheduler_outputs"
os.makedirs(output_dir, exist_ok=True)

# Set guidance scale and number of inference steps (as recommended in model card)
guidance_scale = 4
num_inference_steps = 20

def generate_images_with_schedulers():
    for scheduler_name, scheduler_class in schedulers:
        # Use the selected scheduler
        try:
            pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
        except KeyError as e:
            print(f"Failed to set scheduler {scheduler_name} due to error: {e}")
            continue
        
        # Fix for the warning message
        if hasattr(pipe.scheduler.config, 'lower_order_final'):
            pipe.scheduler.config.lower_order_final = True
        
        # Generate the image
        start_time = time.time()
        try:
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                width=1024,
                height=1024
            )
            image = result.images[0]
        except Exception as e:
            print(f"Error generating image with {scheduler_name}: {e}")
            continue
        
        time_taken = time.time() - start_time
        
        # Save the image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_image_name = f"{scheduler_name.replace(' ', '_')}_{timestamp}.png"
        output_image_path = os.path.join(output_dir, output_image_name)
        image.save(output_image_path)
        print(f"Generated image with {scheduler_name} saved as {output_image_path} (time taken: {time_taken:.2f}s)")

# Run the function to generate images
generate_images_with_schedulers()