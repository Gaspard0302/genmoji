# finetune_model.py
import os
import json
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model
from huggingface_hub import login
import numpy as np
import gc

class EmojiDataset(Dataset):
    """Dataset for emoji images and their descriptions"""
    def __init__(self, image_dir, tokenizer, max_length=77):
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Print the current directory and check if the image_dir exists
        print(f"Current working directory: {os.getcwd()}")
        print(f"Looking for training data in: {self.image_dir.absolute()}")
        print(f"Directory exists: {self.image_dir.exists()}")
        
        # List all files in the directory to debug
        if self.image_dir.exists():
            print(f"Files in directory: {list(self.image_dir.glob('*'))}")
        
        # More flexible file pattern matching
        self.image_paths = sorted([p for p in self.image_dir.glob("*.png")])
        self.text_paths = sorted([p for p in self.image_dir.glob("*.txt")])
        
        print(f"Found {len(self.image_paths)} image files")
        print(f"Found {len(self.text_paths)} text files")
        
        # Match text files to image files by base name
        valid_pairs = []
        for img_path in self.image_paths:
            img_stem = img_path.stem  # Get filename without extension
            txt_path = self.image_dir / f"{img_stem}.txt"
            if txt_path.exists():
                valid_pairs.append((img_path, txt_path))
        
        self.valid_pairs = valid_pairs
        print(f"Found {len(self.valid_pairs)} valid image-text pairs")
        
        # Ensure we have at least one valid pair
        assert len(self.valid_pairs) > 0, "No valid image-text pairs found in the directory"
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        image_path, text_path = self.valid_pairs[idx]
        
        # Load image and convert to RGB
        image = Image.open(image_path).convert("RGB")
        image = image.resize((512, 512))  # Resize to SDXL expected resolution
        image = (torch.from_numpy(np.array(image)) / 255.0).permute(2, 0, 1)
        
        # Load description
        with open(text_path, "r") as f:
            description = f.read().strip()
        
        # Tokenize text
        tokenized_text = self.tokenizer(
            description,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": image.float(),
            "input_ids": tokenized_text.input_ids[0],
            "attention_mask": tokenized_text.attention_mask[0]
        }

def generate_sample_images(pipeline, checkpoint_dir, step):
    """Generate sample images to track model progress"""
    # Create directory for checkpoint images
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set seed for reproducibility
    if pipeline.device.type == "mps":
        # MPS doesn't support generator API the same way
        torch.manual_seed(42)
        generator = None
    else:
        generator = torch.Generator(device=pipeline.device).manual_seed(42)
    
    # Generate smiley face emoji
    smiley_prompt = "emoji of a smiley face"
    smiley_image = pipeline(
        prompt=smiley_prompt,
        num_inference_steps=30,
        generator=generator
    ).images[0]
    
    # Generate unconditional image
    unconditional_image = pipeline(
        prompt="",
        num_inference_steps=30,
        generator=generator
    ).images[0]
    
    # Save images
    smiley_path = os.path.join(checkpoint_dir, f"step_{step}_smiley.png")
    unconditional_path = os.path.join(checkpoint_dir, f"step_{step}_unconditional.png")
    
    smiley_image.save(smiley_path)
    unconditional_image.save(unconditional_path)
    
    print(f"Generated sample images at step {step}")

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SDXL-Lightning for emoji generation")
    parser.add_argument("--model_name", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="Model to fine-tune")
    parser.add_argument("--data_dir", type=str, default="training_data", help="Directory with training data")
    parser.add_argument("--output_dir", type=str, default="emoji_model", help="Output directory for fine-tuned model")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face token for accessing models")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--save_steps", type=int, default=500, help="Save model every X steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--use_mps", action="store_true", default=True, help="Use MPS (Metal) backend on MacOS")
    return parser.parse_args()

def train():
    args = parse_args()
    
    # Make sure the training_data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Warning: Training data directory '{args.data_dir}' does not exist!")
        print("Available directories in current location:")
        print(os.listdir('.'))

    
    # Login to Hugging Face
    login(token=args.hf_token)
    
    # Set device properly for Apple Silicon
    if args.use_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal) backend for acceleration")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {device} device")
    
    # Initialize accelerator - use fp16 only if not on MPS, as MPS has issues with fp16
    use_fp16 = device.type != "mps"
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16" if use_fp16 else "no",
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    # Create directory for checkpoint images
    checkpoint_img_dir = "model_checkpoints"
    os.makedirs(checkpoint_img_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        args.model_name,
        subfolder="tokenizer",
        token=args.hf_token
    )
    
    # Create dataset and dataloader
    dataset = EmojiDataset(args.data_dir, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,  # Lower number of workers to reduce memory pressure
        pin_memory=True  # Enable pin_memory for faster data transfer
    )
    
    # Load model components with memory optimization
    print("Loading model components...")
    
    # Memory optimization - clear cache before loading large models
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_name,
        subfolder="text_encoder",
        token=args.hf_token,
        torch_dtype=torch.float16 if use_fp16 else torch.float32
    )
    
    # Memory optimization - clear cache between model loads
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    
    vae = AutoencoderKL.from_pretrained(
        args.model_name,
        subfolder="vae",
        token=args.hf_token,
        torch_dtype=torch.float16 if use_fp16 else torch.float32
    )
    
    # Memory optimization - clear cache between model loads
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    
    # Load pipeline for UNet - load in half precision to save memory
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.model_name,
        text_encoder=text_encoder,
        vae=vae,
        token=args.hf_token,
        torch_dtype=torch.float16 if use_fp16 else torch.float32,
        low_cpu_mem_usage=True
    )
    unet = pipeline.unet
    
    # Freeze VAE and text encoder to reduce memory requirements
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Set up LoRA for UNet - use smaller rank for memory efficiency
    lora_config = LoraConfig(
        r=args.lora_rank,  # Lower rank for M3 MacBook
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,  # Lower dropout to reduce memory usage
        bias="none"
    )
    
    # Apply LoRA to UNet
    unet = get_peft_model(unet, lora_config)
    unet.train()
    
    # Set up optimizer with weight decay to improve training stability
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01  # Add weight decay for better stability
    )
    
    # Set up learning rate scheduler
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=int(0.05 * len(dataloader) * args.num_epochs),  # 5% warmup
        num_training_steps=len(dataloader) * args.num_epochs
    )
    
    # Set up noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.model_name,
        subfolder="scheduler"
    )
    
    # Prepare for training
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )
    
    # Move models to device
    vae = vae.to(accelerator.device)
    text_encoder = text_encoder.to(accelerator.device)
    
    # Training loop
    global_step = 0
    
    # Track loss for each epoch
    epoch_losses = []
    
    for epoch in range(args.num_epochs):
        progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch+1}/{args.num_epochs}")
        
        epoch_loss = 0
        
        for batch in dataloader:
            # Convert images to latent space
            with torch.no_grad():
                pixel_values = batch["pixel_values"].to(accelerator.device)
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
            
            # Generate random noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), 
                device=accelerator.device
            ).long()
            
            # Add noise to the latents according to the noise magnitude at each timestep
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            

            # Get the text embedding for conditioning
            with torch.no_grad():
                # Get encoder_hidden_states from the text encoder
                encoder_hidden_states = text_encoder(
                    batch["input_ids"].to(accelerator.device)
                )[0]
                
                # IMPORTANT: Resize the encoder hidden states to match what SDXL expects
                # SDXL expects hidden states with dimension 2048 instead of 768
                # We'll use a simple projection to match dimensions
                if encoder_hidden_states.shape[-1] != 2048:
                    # Create a projection layer if it doesn't exist yet
                    if not hasattr(train, 'projection_layer'):
                        train.projection_layer = torch.nn.Linear(
                            encoder_hidden_states.shape[-1], 2048
                        ).to(accelerator.device)
                    
                    # Project the hidden states to the correct dimension
                    encoder_hidden_states = train.projection_layer(encoder_hidden_states)
                
                # Add the required additional conditioning for SDXL
                batch_size = encoder_hidden_states.shape[0]
                text_embeds = torch.ones(
                    (batch_size, 1280),
                    device=accelerator.device,
                    dtype=encoder_hidden_states.dtype
                )
                time_ids = torch.ones(
                    (batch_size, 6),
                    device=accelerator.device,
                    dtype=encoder_hidden_states.dtype
                )
            
            # Predict the noise residual with the required added_cond_kwargs
            noise_pred = unet(
                noisy_latents, 
                timesteps, 
                encoder_hidden_states,
                added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids}
            ).sample
            
            
            # Calculate loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            # Backpropagation
            accelerator.backward(loss)
            
            # Update parameters
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Track loss for this epoch
            epoch_loss += loss.detach().item()
            
            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.detach().item())
            
            global_step += 1
            
            # Explicit memory cleanup after each batch
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()
            
            # Save checkpoint and generate example images
            if global_step % args.save_steps == 0:
                # Save model checkpoint
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                print(f"Saved state to {save_path}")
                
                # Create a pipeline with current model weights for inference
                if accelerator.is_main_process:
                    # Memory cleanup before inference
                    if device.type == "mps":
                        torch.mps.empty_cache()
                    elif device.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Get unwrapped UNet
                    unwrapped_unet = accelerator.unwrap_model(unet)
                    
                    # Create a copy of the pipeline with the current fine-tuned UNet
                    inference_pipeline = StableDiffusionXLPipeline.from_pretrained(
                        args.model_name,
                        unet=unwrapped_unet,
                        text_encoder=text_encoder,
                        vae=vae,
                        token=args.hf_token,
                        torch_dtype=torch.float16 if use_fp16 else torch.float32
                    )
                    
                    # Move to device
                    inference_pipeline = inference_pipeline.to(accelerator.device)
                    
                    # Generate and save example images
                    generate_sample_images(inference_pipeline, checkpoint_img_dir, global_step)
                    
                    # Memory cleanup after inference
                    del inference_pipeline
                    if device.type == "mps":
                        torch.mps.empty_cache()
                    elif device.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
        
        # Calculate average loss for this epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        epoch_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.6f}")
        
        # Memory cleanup at the end of epoch
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    
    # Save the final model
    accelerator.wait_for_everyone()
    unwrapped_unet = accelerator.unwrap_model(unet)
    
    # Save LoRA weights
    unwrapped_unet.save_pretrained(os.path.join(args.output_dir, "final_model"))
    
    # Generate final sample images
    if accelerator.is_main_process:
        # Memory cleanup before final inference
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        # Create a pipeline with final model weights
        final_pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.model_name,
            unet=unwrapped_unet,
            text_encoder=text_encoder,
            vae=vae,
            token=args.hf_token,
            torch_dtype=torch.float16 if use_fp16 else torch.float32
        )
        final_pipeline = final_pipeline.to(accelerator.device)
        
        # Generate and save final example images
        generate_sample_images(final_pipeline, checkpoint_img_dir, "final")
        
        # Save training loss history
        loss_path = os.path.join(args.output_dir, "training_loss.json")
        with open(loss_path, "w") as f:
            json.dump({"epoch_losses": epoch_losses}, f)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    train()