import model_loader
import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch
import gc

# Function to clear GPU memory
def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

# Function to move models to the specified device
def move_models_to_device(models, device):
    for key, model in models.items():
        models[key] = model.to(device)
    return models

# Optimized function to load models
def optimized_model_loader(model_file, device):
    clear_gpu_memory()
    models = model_loader.preload_models_from_standard_weights(model_file, device)
    return models

DEVICE = "cpu"

ALLOW_CUDA = True  # Set this to True to allow CUDA
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

# Clear GPU memory before loading the model
clear_gpu_memory()

# Load tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
model_file = r"data\v1-5-pruned-emaonly.ckpt"

# Try loading the model with optimizations
try:
    models = optimized_model_loader(model_file, DEVICE)  # Load models with optimization
except torch.cuda.OutOfMemoryError:
    print("CUDA out of memory. Try reducing the model size or use a different device.")
    exit()

# Define the prompts
prompt = "A cat stretching on the floor, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
uncond_prompt = ""  # Also known as negative prompt
do_cfg = True
cfg_scale = 8  # min: 1, max: 14

# Image to Image
input_image = None
# Uncomment the following lines to enable image-to-image functionality
# image_path = "../images/dog.jpg"
# input_image = Image.open(image_path)
strength = 0.9  # Higher values mean more noise will be added to the input image

# Sampler configuration
sampler = "ddpm"
num_inference_steps = 30  # Reduced number of steps to save memory
seed = 42

# Generate the output image
try:
    with torch.cuda.amp.autocast():
        output_image = pipeline.generate(
            prompt=prompt,
            uncond_prompt=uncond_prompt,
            input_image=input_image,
            strength=strength,
            do_cfg=do_cfg,
            cfg_scale=cfg_scale,
            sampler_name=sampler,
            n_inference_steps=num_inference_steps,
            seed=seed,
            models=models,
            device=DEVICE,
            idle_device="cpu",
            tokenizer=tokenizer,
        )
except torch.cuda.OutOfMemoryError:
    print("CUDA out of memory during generation. Consider reducing image size or number of steps.")
    exit()

# Save and display the output image
output_image_pil = Image.fromarray(output_image)
output_image_pil.show()
