from grat import GratFluxAttnProcessor, init_local_mask_flex
from diffusers import FluxPipeline
import torch 

model_id = "black-forest-labs/FLUX.1-dev"
pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
height = 1024
width = 2048
device = torch.device('cuda')
prompt = "Crystal hummingbirds drinking from floating amethyst flowers in misty dawn garden, iridescent wings motion-blurred, dewdrops scattering rainbow caustics, dreamy macro fantasy realism"

attenable = len(pipe.tokenizer(prompt)['input_ids'])
group_h, group_w = 16, 16
text_length = 512  

mask = init_local_mask_flex(
    height // 16, width // 16, text_length=text_length, attenable_text=attenable,
    group_h=group_h, group_w=group_w, device=device
)
attn_processors = {}
for k, v in pipe.transformer.attn_processors.items():
    if "single_transformer_blocks" in k:
        attn_processors[k] = v
    else:
        attn_processors[k] = GratFluxAttnProcessor(
            mask, height // 16, width // 16, group_h, group_w, text_length
        )
pipe.transformer.set_attn_processor(attn_processors)

output = pipe(prompt=prompt, height=height, width=width, num_inference_steps=28, max_sequence_length=text_length).images[0]
