import torch
from dalle2_pytorch import DiffusionPrior, DiffusionPriorNetwork, OpenAIClipAdapter
from dalle2_pytorch import Unet, Decoder,DALLE2
import json

with open('weights/prior_config.json', 'r') as f:
    prior_config = json.load(f)

prior_config = prior_config["prior"]["net"]
prior_network = DiffusionPriorNetwork(**prior_config)


diffusion_prior = DiffusionPrior(
    net=prior_network,
    clip=OpenAIClipAdapter("ViT-L/14"),
    image_embed_dim=768,
    timesteps=1000,
    sample_timesteps = 100,
    cond_drop_prob=0.1,
    loss_type="l2",
    condition_on_text_encodings=True,

)

diffusion_prior.load_state_dict(torch.load("weights/prior_best.pth",map_location=torch.device('cpu')),strict=True)



with open('weights/decoder_config.json', 'r') as f:
    decoder_config = json.load(f)

unet_config = decoder_config["decoder"]["unets"][0]
unet = Unet(**unet_config)


decoder = Decoder(
    unet = unet,
    clip=OpenAIClipAdapter("ViT-L/14"),
    timesteps = 1000,
    image_sizes = [224],
    image_cond_drop_prob = 0.1,
    text_cond_drop_prob = 0.5,
    sample_timesteps = 100,
    learned_variance=True,
)

decoder.load_state_dict(torch.load("weights/decoder_best.pth",map_location=torch.device('cpu')),strict=True)

dalle2 = DALLE2(
    prior = diffusion_prior,
    decoder = decoder
)


images = dalle2(
    ['a child eating a birthday cake near some balloons'],
    cond_scale = 2., # classifier free guidance strength (> 1 would strengthen the condition)
    return_pil_images=True,
)
for img in images:
    img.save("out.jpg")