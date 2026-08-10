import argparse
import torch
from sana_model import SanaMS_1600M_P1_D20, load_pth

# ---- fixed SANA 1.6B / 1024px settings (from the official config) ----
TRANSFORMER_CKPT = "SANA1.5_1.6B_1024px.pth"          # local path to the official .pth
DC_AE_REPO       = "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers"
GEMMA_REPO       = "Efficient-Large-Model/gemma-2-2b-it"  # open mirror of google/gemma-2-2b-it
SCALING_FACTOR   = 0.41407     # DC-AE latent scale
FLOW_SHIFT       = 3.0         # FlowMatchEuler shift for 1024px
MAX_LEN          = 300         # model_max_length
VAE_DOWNSAMPLE   = 32          # f32 -> latent is image/32

# The "complex human instruction" prepended to every prompt (verbatim from the config).
# It steers Gemma to emit an enhanced visual description; SANA was tuned to use it.
CHI_PROMPT = "\n".join([
    'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:',
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
    "User Prompt: ",
])


# --------------------------------------------------------------------------- #
# text encoding
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_prompt(prompt, tokenizer, text_encoder, device, dtype):
    """
    Returns (emb, mask):
      emb:  (1, 300, 2304)  Gemma hidden states, BOS + last 299 positions
      mask: (1, 300)        1 = real token, 0 = padding
    """
    full = CHI_PROMPT + prompt
    # pad to (chi tokens + 300 - 2); the -2 accounts for [bos] and a leading space token
    num_chi_tokens = len(tokenizer.encode(CHI_PROMPT))
    max_length_all = num_chi_tokens + MAX_LEN - 2

    tok = tokenizer([full], max_length=max_length_all, padding="max_length",
                    truncation=True, return_tensors="pt").to(device)

    # keep the BOS embedding (index 0) plus the final 299 positions — the tail holds
    # the user prompt's content after the long CHI prefix has been consumed as context.
    select_index = [0] + list(range(-MAX_LEN + 1, 0))
    hidden = text_encoder(input_ids=tok.input_ids,
                          attention_mask=tok.attention_mask)[0]   # (1, L, 2304)
    emb = hidden[:, select_index].to(dtype)                       # (1, 300, 2304)
    mask = tok.attention_mask[:, select_index]                    # (1, 300)
    return emb, mask


@torch.no_grad()
def encode_null(tokenizer, text_encoder, device, dtype):
    """Unconditional embedding: encode the empty string, already exactly 300 tokens."""
    tok = tokenizer("", max_length=MAX_LEN, padding="max_length",
                    truncation=True, return_tensors="pt").to(device)
    hidden = text_encoder(input_ids=tok.input_ids,
                          attention_mask=tok.attention_mask)[0]   # (1, 300, 2304)
    mask = tok.attention_mask                                     # (1, 300)
    return hidden.to(dtype), mask


# --------------------------------------------------------------------------- #
# the denoising loop
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate(model, scheduler, cond, cond_mask, null, null_mask,
             latent_hw, steps, guidance_scale, device, dtype, seed=0):
    """
    Flow-matching Euler sampling with classifier-free guidance.
    `model(x, t, y, mask)` returns the flow velocity (pred_sigma=False -> 32 channels).
    """
    h, w = latent_hw
    g = torch.Generator(device=device).manual_seed(seed)

    # pure Gaussian noise = the flow-matching prior at t=1. Latents kept in fp32.
    z = torch.randn(1, 32, h, w, generator=g, device=device, dtype=torch.float32)

    do_cfg = guidance_scale > 1.0
    if do_cfg:
        emb = torch.cat([null, cond], dim=0)             # (2, 300, 2304)
        mask = torch.cat([null_mask, cond_mask], dim=0)  # (2, 300)
    else:
        emb, mask = cond, cond_mask

    scheduler.set_timesteps(steps, device=device)
    for t in scheduler.timesteps:
        latent_in = torch.cat([z, z], dim=0) if do_cfg else z
        t_batch = t.expand(latent_in.shape[0])

        # model runs in `dtype`; feed it that, get velocity back in fp32 for a stable step
        v = model(latent_in.to(dtype), t_batch, emb, mask).float()

        if do_cfg:
            v_uncond, v_text = v.chunk(2, dim=0)
            v = v_uncond + guidance_scale * (v_text - v_uncond)   # CFG

        z = scheduler.step(v, t, z, return_dict=False)[0]         # Euler integration step

    return z  # (1, 32, h, w) clean latent


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, default="A beautiful landscape with mountains and a river")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance_scale", type=float, default=4.5)
    ap.add_argument("--image_size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sana_out.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"device={device} dtype={dtype}")

    # ---- load the three models ----
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from diffusers import AutoencoderDC, FlowMatchEulerDiscreteScheduler
    from torchvision.utils import save_image

    print("loading Gemma-2 ...")
    tokenizer = AutoTokenizer.from_pretrained(GEMMA_REPO)
    tokenizer.padding_side = "right"
    text_encoder = (AutoModelForCausalLM.from_pretrained(GEMMA_REPO, torch_dtype=dtype)
                    .get_decoder().to(device).eval())

    print("loading DC-AE decoder ...")
    dc_ae = AutoencoderDC.from_pretrained(DC_AE_REPO, torch_dtype=dtype).to(device).eval()
    scaling_factor = getattr(dc_ae.config, "scaling_factor", SCALING_FACTOR) or SCALING_FACTOR

    print("loading SANA transformer ...")
    model = SanaMS_1600M_P1_D20().to(device).to(dtype).eval()
    missing, unexpected = load_pth(model, TRANSFORMER_CKPT)
    print(f"  transformer loaded (missing={len(missing)}, unexpected={len(unexpected)})")

    scheduler = FlowMatchEulerDiscreteScheduler(shift=FLOW_SHIFT)

    # ---- encode text ----
    print("encoding prompt ...")
    cond, cond_mask = encode_prompt(args.prompt, tokenizer, text_encoder, device, dtype)
    null, null_mask = encode_null(tokenizer, text_encoder, device, dtype)

    # ---- sample ----
    latent = args.image_size // VAE_DOWNSAMPLE
    print(f"sampling {args.steps} steps at latent {latent}x{latent} ...")
    z = generate(model, scheduler, cond, cond_mask, null, null_mask,
                 (latent, latent), args.steps, args.guidance_scale, device, dtype, args.seed)

    # ---- decode ----
    print("decoding ...")
    image = dc_ae.decode((z / scaling_factor).to(dtype)).sample   # (1, 3, H, W) in [-1, 1]
    image = (image.float() / 2 + 0.5).clamp(0, 1)
    save_image(image, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
