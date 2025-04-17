import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

def show():
    # Streamlit UI
    st.title("🎨 AI Image Generator")
    prompt = st.text_input("Describe your image (e.g., 'a cyberpunk cat')")
    creativity = st.slider("Creativity (higher = more random)", 0.0, 1.0, 0.7)

    if st.button("Generate Image"):
        if prompt:
            with st.spinner("✨ Generating your image..."):
                # Load Hugging Face Stable Diffusion
                pipe = StableDiffusionPipeline.from_pretrained(
                    "prompthero/openjourney",  # 2x smaller than runwayml/stable-diffusion-v1-5
                    torch_dtype=torch.float32
                ).to("cuda" if torch.cuda.is_available() else "cpu")

                # Generate image
                image = pipe(
                    prompt, 
                    guidance_scale=7.5,  # Controls creativity (higher = more diverse)
                    height=512, width=512, # Limitting height and width to decrease memory usage
                    num_inference_steps=50
                ).images[0]

                st.image(image, caption=f"Generated: '{prompt}'")
        else:
            st.warning("Please enter a prompt!")
