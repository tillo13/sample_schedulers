# Scheduler Sampling Scripts

These scripts are designed to demonstrate the use of various diffusion schedulers with Stable Diffusion models. They generate images using different scheduling algorithms to show how each scheduler affects the outcome.

## Overview

Diffusion models generate images iteratively by denoising data across multiple steps. Schedulers play a crucial role in controlling the denoising process. Different schedulers can impact the quality, style, and speed of image generation. These scripts allow you to explore and compare the effects of various schedulers.

### What are Schedulers?

Schedulers are algorithms that determine how noise is added or removed at each step during the diffusion process. They guide the denoising process to ensure that the final output image closely matches the desired prompt. The schedulers used in these scripts are:

- **LMSDiscreteScheduler**
- **DPMSolverMultistepScheduler**
- **DDIMScheduler**
- **EulerDiscreteScheduler**
- **PNDMScheduler**
- **EulerAncestralDiscreteScheduler**
- **HeunDiscreteScheduler**
- **DEISMultistepScheduler**
- **DPMSolverSinglestepScheduler**
- **KDPM2DiscreteScheduler**
- **KDPM2AncestralDiscreteScheduler**
- **UniPCMultistepScheduler**

## Scripts

### 1. `sample_text2image_scheduler.py`

This script generates images solely from text prompts using the `ProteusV0.4` model. It loops through different schedulers, generates an image for each scheduler, and saves the results.

#### How it works:
1. **Load Model**: The `StableDiffusionXLPipeline` model is loaded along with a VAE.
2. **Disable Safety Checker**: For this example, the safety checker is disabled.
3. **Set Prompts**: Define the primary text prompt and a negative prompt to guide the image generation.
4. **Set Schedulers**: An array of scheduler classes is defined.
5. **Generate Images**:
   - For each scheduler, the script sets the scheduler configuration and generates an image based on the prompt.
   - The generated image is saved with a timestamp and scheduler name for easy identification.
   - The time taken for each generation is recorded and printed.

```python
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
negative_prompt = "bad quality, bad anatomy, worst quality, low quality, low resolutions, extra fingers, blur, blurry, ugly, wrongs proportions, watermark, image artifacts, lowres, ugly, jpeg artifacts, deformed, noisy image"

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
2. sample_schedulers_with_input.py
This script generates images based on a given input image using the Ghibli Diffusion model. It loops through different schedulers, generates an image for each scheduler, and saves the results.

How it works:
Load Model: The StableDiffusionImg2ImgPipeline model is loaded.
Disable Safety Checker: For this example, the safety checker is disabled.
Set Prompts: Define the primary text prompt and a negative prompt to guide the image generation.
Load Input Image: An initial image is loaded and resized to the preferred input size.
Set Schedulers: An array of scheduler classes is defined.
Generate Images:
For each scheduler, the script sets the scheduler configuration and generates an image based on the prompt and initial image.
The generated image is saved with a timestamp and scheduler name for easy identification.
The time taken for each generation is recorded and printed.
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
Running the Scripts
Activate your virtual environment:

On Windows:

.\hidiff_env\Scripts\activate
On macOS and Linux:

source hidiff_env/bin/activate
Run the scripts:

For sample_text2image_scheduler.py:

python sample_text2image_scheduler.py
For sample_schedulers_with_input.py:

python sample_schedulers_with_input.py
This should generate images based on the specified models and prompts, demonstrating the effects of different schedulers.

This README provides a comprehensive explanation of the functionality and purpose of each script, making it accessible for novice users to understand and use effectively.