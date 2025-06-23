import torch
from torch import nn
import comfy.sd
import folder_paths
from comfy import model_management

MAX_RESOLUTION=16384

class Downsample2D(nn.Module):
    def __init__(self):
        super().__init__()
        # original conv has stride = 2
        self.conv = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding="same")

    def forward(self, hidden_states, *args, **kwargs):
        return self.conv(hidden_states)

class Upsample2D(nn.Module):
    def __init__(self):
        super().__init__()
        # original layer has an F.Interpolate 2x upsampling operation
        self.conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding="same")

    def forward(self, hidden_states, output_size=None, *args, **kwargs):
        return self.conv(hidden_states)


class ModifiedSDXLVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae"),),
                "to_modify": (["both", "decoder", "encoder"],),
            },
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_modified_vae"
    CATEGORY = "loaders/modified"

    def load_modified_vae(self, vae_name, to_modify):
        vae_path = folder_paths.get_full_path("vae", vae_name)
        sd = comfy.utils.load_torch_file(vae_path)
        vae = comfy.sd.VAE(sd=sd)
        model = vae.first_stage_model

        device = model_management.vae_device()
        working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
        dtype = model_management.vae_dtype(device, working_dtypes)

        with torch.no_grad():
            if to_modify in {"encoder", "both"}:
                mockdown = Downsample2D().to(device=device, dtype=dtype)

                # diffusers
                # target_downsampler = model.encoder.down_blocks[0].downsamplers[0]
                # mockdown.conv.weight.copy_(target_downsampler.conv.weight)
                # mockdown.conv.bias.copy_(target_downsampler.conv.bias)
                # model.encoder.down_blocks[0].downsamplers[0] = mockdown

                target_downsampler = model.encoder.down[0].downsample

                mockdown.conv.weight.copy_(target_downsampler.conv.weight)
                mockdown.conv.bias.copy_(target_downsampler.conv.bias)
                model.encoder.down[0].downsample = mockdown

            if to_modify in {"decoder", "both"}:
                mockup = Upsample2D().to(device=device, dtype=dtype)

                # diffusers
                # target_upsampler = model.decoder.up_blocks[2].upsamplers[0]
                # mockup.conv.weight.copy_(target_upsampler.conv.weight)
                # mockup.conv.bias.copy_(target_upsampler.conv.bias)
                # # original layer has some sort of upsampling which gets removed
                # model.decoder.up_blocks[2].upsamplers[0] = mockup

                target_upsampler = model.decoder.up[1].upsample
                mockup.conv.weight.copy_(target_upsampler.conv.weight)
                mockup.conv.bias.copy_(target_upsampler.conv.bias)
                model.decoder.up[1].upsample = mockup

        model.to(device)
        return (vae,)

class EmptyLatentImageVariable:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The width of the latent images in pixels."}),
                "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The height of the latent images in pixels."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "The number of latent images in the batch."}),
                "channels": ("INT", {"default": 4, "min": 1, "max": 256, "step": 1, "tooltip": "Channels of the latent, 4 for sdxl vae and 16 for flux vae."}),
                "compression": ("INT", {"default": 8, "min": 2, "max": 256, "step": 2, "tooltip": "The latent compression, most models use 8."}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The empty latent image batch.",)
    FUNCTION = "generate"

    CATEGORY = "latent"
    DESCRIPTION = "Create a new batch of empty latent images to be denoised via sampling."

    def generate(self, width, height, batch_size=1, channels=4, compression=8):
        latent = torch.zeros([batch_size, channels, height // compression, width // compression], device=self.device)
        return ({"samples":latent}, )

NODE_CLASS_MAPPINGS = {
    "ModifiedSDXLVAELoader": ModifiedSDXLVAELoader,
    "EmptyLatentImageVariable": EmptyLatentImageVariable,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModifiedSDXLVAELoader": "Modified SDXL VAE Loader",
    "EmptyLatentImageVariable": "Empty Latent Image Variable",
}