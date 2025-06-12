import torch
from torch import nn
import comfy.sd
import folder_paths


class Downsample2d(nn.Module):
    def __init__(self):
        super().__init__()
        # original conv has stride = 2
        self.conv = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding="same")

    def forward(self, hidden_states, *args, **kwargs):
        return self.conv(hidden_states)

class Upsample2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding="same")

    def forward(self, hidden_states, output_size=None, *args, **kwargs):
        return self.conv(hidden_states)


class ModifiedSDXLVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae"), ),
            },
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_modified_vae"
    CATEGORY = "loaders/modified"

    def load_modified_vae(self, vae_name):
        vae_path = folder_paths.get_full_path("vae", vae_name)
        sd = comfy.utils.load_torch_file(vae_path)
        vae = comfy.sd.VAE(sd=sd)
        model = vae.first_stage_model

        p = next(model.parameters())
        device, dtype = p.device, p.dtype

        with torch.no_grad():
            mockdown = Downsample2d().to(device=device, dtype=dtype)
            mockup = Upsample2D().to(device=device, dtype=dtype)

            # diffusers
            # target_downsampler = model.encoder.down_blocks[0].downsamplers[0]
            # mockdown.conv.weight.copy_(target_downsampler.conv.weight)
            # mockdown.conv.bias.copy_(target_downsampler.conv.bias)
            # model.encoder.down_blocks[0].downsamplers[0] = mockdown

            target_downsampler = model.encoder.down[0].downsample



            mockdown.conv.weight.copy_(target_downsampler.conv.weight)
            mockdown.conv.bias.copy_(target_downsampler.conv.bias)
            model.encoder.down[0].downsample = mockdown

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
