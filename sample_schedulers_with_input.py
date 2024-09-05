import os
import time
import torch
from datetime import datetime
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.schedulers import (
    LMSDiscreteScheduler, DPMSolverMultistepScheduler, DDIMScheduler,
    EulerDiscreteScheduler, PNDMScheduler, EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler, DEISMultistepScheduler, DPMSolverSinglestepScheduler,
    KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler,
    UniPCMultistepScheduler
)

# Load the Ghibli Diffusion model
model_id = "nitrosocke/Ghibli-Diffusion"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Disable safety checker
pipe.safety_checker = None

# Define the prompt and the negative prompt
prompt = (
    "ghibli style portrait of a family, with detailed facial features such as eyes, nose, lips, "
    "and hair, in the style of Studio Ghibli. Realistic shading, expressive eyes, clear skin texture."
)
negative_prompt = "blurry, ugly, duplicate, poorly drawn face, deformed, mosaic, artifacts, bad limbs"

# Seed for reproducibility
seed = 3450349066
torch.manual_seed(seed)  # Set the global seed for reproducibility
generator = torch.Generator(device="cuda").manual_seed(seed)  # Create a reproducible generator

# Load the input image
input_image_path = "./451.jpg"
init_image = Image.open(input_image_path).convert("RGB")

# Ensure image is loaded correctly
assert init_image is not None, "Failed to load input image."

# Resize the image to be square and match Diffusion's preferred input size
init_image = init_image.resize((512, 512))

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
output_dir = "./scheduler_outputs"
os.makedirs(output_dir, exist_ok=True)

# Set the strength and guidance scale
strength = 0.6
guidance_scale = 8.0

def generate_images_with_schedulers():
    for scheduler_name, scheduler_class in schedulers:
        # Use the selected scheduler
        pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
        
        # Fix for the warning message
        if hasattr(pipe.scheduler.config, 'lower_order_final'):
            pipe.scheduler.config.lower_order_final = True
        
        # Generate the image
        start_time = time.time()
        result = pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=100,
            negative_prompt=negative_prompt,
            generator=generator
        )
        image = result.images[0]
        time_taken = time.time() - start_time
        
        # Save the image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_image_name = f"ghibli_{scheduler_name.replace(' ', '_')}_{timestamp}.png"
        output_image_path = os.path.join(output_dir, output_image_name)
        image.save(output_image_path)
        print(f"Generated image with {scheduler_name} saved as {output_image_path} (time taken: {time_taken:.2f}s)")

# Run the function to generate images
generate_images_with_schedulers()