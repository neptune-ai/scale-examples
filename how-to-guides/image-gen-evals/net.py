import torch
import torchvision
from tempfile import NamedTemporaryFile
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from tqdm.auto import tqdm, trange
from diffusers import DiffusionPipeline
import os
import json
from typing import List, Optional, Union, Tuple, Dict, Any
import numpy as np
from PIL import Image


def get_device() -> str:
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    return device


class ClassConditionedUnet(nn.Module):
    def __init__(self, num_classes=10, class_emb_size=4):
        super().__init__()

        # The embedding layer will map the class label to a vector of size class_emb_size
        self.class_emb = nn.Embedding(num_classes, class_emb_size)

        # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
        self.model = UNet2DModel(
            sample_size=28,           # the target image resolution
            in_channels=1 + class_emb_size, # Additional input channels for class cond.
            out_channels=1,           # the number of output channels
            layers_per_block=2,       # how many ResNet layers to use per UNet block
            block_out_channels=(32, 64, 64),
            down_block_types=(
                "DownBlock2D",        # a regular ResNet downsampling block
                "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",          # a regular ResNet upsampling block
            ),
        )

    # Our forward method now takes the class labels as an additional argument
    def forward(self, x, t, class_labels):
        # Shape of x:
        bs, ch, w, h = x.shape

        # class conditioning in right shape to add as additional input channels
        class_cond = self.class_emb(class_labels) # Map to embedding dimension
        class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)
        # x is shape (bs, 1, 28, 28) and class_cond is now (bs, 4, 28, 28)

        # Net input is now x and class cond concatenated together along dimension 1
        net_input = torch.cat((x, class_cond), 1) # (bs, 5, 28, 28)

        # Feed this to the UNet alongside the timestep and return the prediction
        return self.model(net_input, t).sample # (bs, 1, 28, 28)
    
    @property
    def dtype(self):
        """Return the dtype of the model parameters."""
        return next(self.parameters()).dtype


class ClassConditionedPipeline(DiffusionPipeline):
    """
    A class-conditioned diffusion pipeline for generating MNIST-like images.
    
    This pipeline extends DiffusionPipeline to support class-conditioned generation,
    training state management, and convenient output handling.
    """
    
    def __init__(self, unet: ClassConditionedUnet, scheduler: DDPMScheduler):
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler
        self.register_modules(unet=unet, scheduler=scheduler)
    
    @torch.no_grad()
    def __call__(
        self,
        class_labels: Union[torch.Tensor, List[int], int],
        batch_size: int = 1,
        num_inference_steps: int = 1000,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
        return_pil: bool = False,
        show_progress: bool = True,
    ) -> Union[Dict[str, Any], Tuple]:
        """
        Generate images conditioned on class labels.
        
        Args:
            class_labels: Class labels for conditioning. Can be:
                - int: Single class label (will be repeated batch_size times)
                - List[int]: List of class labels (length should match batch_size)
                - torch.Tensor: Tensor of class labels
            batch_size: Number of images to generate
            num_inference_steps: Number of denoising steps
            generator: Random generator for reproducibility
            return_dict: Whether to return results as dict
            return_pil: Whether to return PIL images instead of tensors
            show_progress: Whether to show progress bar
            
        Returns:
            Dict with 'images' key containing generated images, or tuple if return_dict=False
        """
        device = self._execution_device
        
        # Handle class labels input
        if isinstance(class_labels, int):
            class_labels = [class_labels] * batch_size
        elif isinstance(class_labels, list):
            if len(class_labels) != batch_size:
                raise ValueError(f"Length of class_labels ({len(class_labels)}) must match batch_size ({batch_size})")
        
        if not isinstance(class_labels, torch.Tensor):
            class_labels = torch.tensor(class_labels, device=device, dtype=torch.long)
        else:
            class_labels = class_labels.to(device)
        
        # Initialize random noise
        shape = (batch_size, 1, 28, 28)  # MNIST image shape
        latents = torch.randn(shape, generator=generator, device=device, dtype=self.unet.dtype)
        
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # Denoising loop
        iterator = tqdm(timesteps, desc="Generating images") if show_progress else timesteps
        for t in iterator:
            # Predict noise residual
            noise_pred = self.unet(latents, t, class_labels)
            
            # Compute previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, generator=generator).prev_sample
        
        # Post-processing
        images = latents.detach().cpu().clamp(-1, 1)
        
        if return_pil:
            # Convert to PIL images
            images_pil = []
            for img in images:
                # Convert from [-1, 1] to [0, 255]
                img_np = ((img[0] + 1) * 127.5).clamp(0, 255).numpy().astype(np.uint8)
                images_pil.append(Image.fromarray(img_np, mode='L'))
            images = images_pil
        
        if return_dict:
            return {"images": images, "class_labels": class_labels.cpu()}
        else:
            return (images,)
    
    def generate_grid(
        self,
        classes_per_row: int = 8,
        num_rows: int = 10,
        num_inference_steps: int = 1000,
        generator: Optional[torch.Generator] = None,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a grid of images with different classes.
        
        Args:
            classes_per_row: Number of images per class (columns)
            num_rows: Number of different classes (rows)
            num_inference_steps: Number of denoising steps
            generator: Random generator for reproducibility
            show_progress: Whether to show progress bar
            
        Returns:
            Dict containing:
                - 'images': Individual images as tensor (batch_size, 1, 28, 28)
                - 'grid': Combined grid image as tensor (1, H, W)
                - 'pil_images': Individual images as PIL Images
                - 'pil_grid': Grid as PIL Image
                - 'class_labels': Class labels used
        """
        batch_size = classes_per_row * num_rows
        class_labels = torch.tensor([[i] * classes_per_row for i in range(num_rows)]).flatten()
        
        # Generate images
        result = self(
            class_labels=class_labels,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            generator=generator,
            return_dict=True,
            return_pil=False,
            show_progress=show_progress,
        )
        
        images = result["images"]
        
        # Create grid
        grid_tensor = torchvision.utils.make_grid(images, nrow=classes_per_row, padding=2, pad_value=1)
        
        # Convert to PIL
        pil_images = []
        for img in images:
            img_np = ((img[0] + 1) * 127.5).clamp(0, 255).numpy().astype(np.uint8)
            pil_images.append(Image.fromarray(img_np, mode='L'))
        
        # Grid as PIL
        grid_np = ((grid_tensor[0] + 1) * 127.5).clamp(0, 255).numpy().astype(np.uint8)
        pil_grid = Image.fromarray(grid_np, mode='L')
        
        return {
            "images": images,
            "grid": grid_tensor,
            "pil_images": pil_images,
            "pil_grid": pil_grid,
            "class_labels": class_labels,
        }
    
    def save_training_state(
        self,
        save_directory: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        step: int = 0,
        loss: float = 0.0,
        **kwargs
    ):
        """
        Save complete training state for resuming training.
        
        Args:
            save_directory: Directory to save the state
            optimizer: Optimizer to save state for
            epoch: Current epoch
            step: Current step
            loss: Current loss value
            **kwargs: Additional state to save
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model components using HF format
        self.save_pretrained(save_directory)
        
        # Save training state
        training_state = {
            "epoch": epoch,
            "step": step,
            "loss": loss,
            **kwargs
        }
        
        if optimizer is not None:
            training_state["optimizer_state_dict"] = optimizer.state_dict()
        
        torch.save(training_state, os.path.join(save_directory, "training_state.pt"))
        
        # Save training config
        config = {
            "num_classes": self.unet.class_emb.num_embeddings,
            "class_emb_size": self.unet.class_emb.embedding_dim,
        }
        
        with open(os.path.join(save_directory, "training_config.json"), "w") as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load_training_state(
        cls,
        load_directory: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[str] = None,
    ) -> Tuple["ClassConditionedPipeline", Dict[str, Any]]:
        """
        Load pipeline and training state from directory.
        
        Args:
            load_directory: Directory containing saved state
            optimizer: Optimizer to load state into
            device: Device to load model on
            
        Returns:
            Tuple of (pipeline, training_state_dict)
        """
        # Load pipeline using HF format
        pipeline = cls.from_pretrained(load_directory)
        
        if device:
            pipeline = pipeline.to(device)
        
        # Load training state
        training_state_path = os.path.join(load_directory, "training_state.pt")
        training_state = {}
        
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location=device)
            
            # Load optimizer state if provided
            if optimizer is not None and "optimizer_state_dict" in training_state:
                optimizer.load_state_dict(training_state["optimizer_state_dict"])
        
        return pipeline, training_state
    
    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save the pipeline components to a directory.
        
        This implementation ensures that the custom UNet model is saved properly.
        """
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the scheduler (this works fine with HF)
        scheduler_path = os.path.join(save_directory, "scheduler")
        self.scheduler.save_pretrained(scheduler_path)
        
        # Save the UNet manually since it's a custom class
        unet_path = os.path.join(save_directory, "unet")
        os.makedirs(unet_path, exist_ok=True)
        
        # Save UNet weights
        torch.save(self.unet.state_dict(), os.path.join(unet_path, "diffusion_pytorch_model.bin"))
        
        # Save UNet config
        unet_config = {
            "num_classes": self.unet.class_emb.num_embeddings,
            "class_emb_size": self.unet.class_emb.embedding_dim,
            "_class_name": "ClassConditionedUnet"
        }
        
        with open(os.path.join(unet_path, "config.json"), "w") as f:
            json.dump(unet_config, f, indent=2)
        
        # Create model index
        model_index = {
            "_class_name": "ClassConditionedPipeline",
            "_diffusers_version": "0.33.1",
            "scheduler": ["diffusers", "DDPMScheduler"],
            "unet": ["net", "ClassConditionedUnet"]
        }
        
        with open(os.path.join(save_directory, "model_index.json"), "w") as f:
            json.dump(model_index, f, indent=2)
        
        print(f"Pipeline saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """
        Load pipeline from pretrained weights.
        
        This method properly loads the custom UNet model.
        """
        import os
        
        # Load model index to get component info
        model_index_path = os.path.join(pretrained_model_name_or_path, "model_index.json")
        if os.path.exists(model_index_path):
            with open(model_index_path, "r") as f:
                model_index = json.load(f)
        else:
            raise FileNotFoundError(f"model_index.json not found in {pretrained_model_name_or_path}")
        
        # Load scheduler
        scheduler_path = os.path.join(pretrained_model_name_or_path, "scheduler")
        if os.path.exists(scheduler_path):
            scheduler = DDPMScheduler.from_pretrained(scheduler_path)
        else:
            print("Warning: No scheduler found, using default")
            scheduler = DDPMScheduler(num_train_timesteps=1000)
        
        # Load UNet config and weights
        unet_path = os.path.join(pretrained_model_name_or_path, "unet")
        unet_config_path = os.path.join(unet_path, "config.json")
        unet_weights_path = os.path.join(unet_path, "diffusion_pytorch_model.bin")
        
        if os.path.exists(unet_config_path):
            with open(unet_config_path, "r") as f:
                unet_config = json.load(f)
                
            num_classes = unet_config.get("num_classes", 10)
            class_emb_size = unet_config.get("class_emb_size", 4)
        else:
            print("Warning: No UNet config found, using defaults")
            num_classes = 10
            class_emb_size = 4
        
        # Create UNet and load weights
        unet = ClassConditionedUnet(num_classes=num_classes, class_emb_size=class_emb_size)
        
        if os.path.exists(unet_weights_path):
            state_dict = torch.load(unet_weights_path, map_location="cpu")
            unet.load_state_dict(state_dict)
            print(f"Loaded UNet weights from {unet_weights_path}")
        else:
            print("Warning: No UNet weights found, using random initialization")
        
        return cls(unet=unet, scheduler=scheduler)
    
    @property
    def _execution_device(self):
        """Get the device where the pipeline should run."""
        return next(self.unet.parameters()).device
