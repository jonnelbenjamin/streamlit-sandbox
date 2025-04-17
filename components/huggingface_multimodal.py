import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import (
    pipeline,
    BlipProcessor, 
    BlipForConditionalGeneration,
    CLIPProcessor, 
    CLIPModel,
    AutoFeatureExtractor,
    AutoModelForAudioClassification
)
import torchaudio
from utils.config import HF_API_KEY

def show():
    st.title("🚀 Multimodal AI Playground")
    st.markdown("### Experiment with cutting-edge cross-modal models")

    # Sidebar with model selection
    with st.sidebar:
        st.header("Configuration")
        task = st.selectbox(
            "Select Mode",
            options=[
                "image_captioning",
                "text_to_image_search",
                "audio_classification",
                "model_comparison",
                "embedding_visualizer"
            ],
            format_func=lambda x: {
                "image_captioning": "📷 Image Captioning",
                "text_to_image_search": "🔍 Text-to-Image Search",
                "audio_classification": "🎵 Audio Analysis",
                "model_comparison": "🤝 Model Comparison",
                "embedding_visualizer": "🧠 Embedding Space"
            }[x]
        )

    # Unique Feature 1: Interactive Image Captioning with Style Control
    if task == "image_captioning":
        st.subheader("📷 Creative Image Captioning")
        col1, col2 = st.columns(2)
        
        with col1:
            img_source = st.radio("Image source", ["Upload", "URL"])
            if img_source == "Upload":
                image_file = st.file_uploader("Upload image", type=["jpg", "png"])
                image = Image.open(image_file) if image_file else None
            else:
                image_url = st.text_input("Image URL", "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png")
                if image_url:
                    try:
                        response = requests.get(image_url)
                        image = Image.open(BytesIO(response.content))
                    except:
                        st.error("Couldn't load image from URL")
            
            if image:
                st.image(image, caption="Input Image", use_column_width=True)

        with col2:
            if image:
                style = st.selectbox(
                    "Caption Style",
                    ["Default", "Humorous", "Technical", "Poetic", "Emoji"],
                    index=0
                )
                
                prompt = {
                    "Default": "a photography of",
                    "Humorous": "a funny caption for this image:",
                    "Technical": "a technical description of",
                    "Poetic": "a poetic caption for",
                    "Emoji": "describe this image using emojis:"
                }[style]
                
                if st.button("Generate Caption"):
                    with st.spinner("Generating creative caption..."):
                        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
                        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
                        
                        inputs = processor(image, prompt, return_tensors="pt")
                        out = model.generate(**inputs, max_new_tokens=50)
                        caption = processor.decode(out[0], skip_special_tokens=True)
                        
                        st.subheader("Generated Caption")
                        st.markdown(f"**{caption}**")
                        st.balloons()

    # Unique Feature 2: Text-to-Image Semantic Search
    elif task == "text_to_image_search":
        st.subheader("🔍 Find Images by Concept")
        
        query = st.text_input("Describe what you want to find:", "a happy corgi playing in the snow")
        num_results = st.slider("Number of results", 1, 9, 3)
        
        if st.button("Search"):
            with st.spinner(f"Finding images matching '{query}'..."):
                # Use CLIP embeddings to find similar images
                model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                
                # Sample dataset (in reality you'd use a proper image dataset)
                sample_images = [
                    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png",
                    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/corgi.png",
                    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/stable_diffusion/stable_diffusion_1.png"
                ]
                
                # Encode query
                inputs = processor(text=[query], return_tensors="pt", padding=True)
                text_features = model.get_text_features(**inputs)
                
                # Compare with images
                results = []
                for img_url in sample_images:
                    try:
                        image = Image.open(BytesIO(requests.get(img_url).content))
                        image_inputs = processor(images=image, return_tensors="pt", padding=True)
                        image_features = model.get_image_features(**image_inputs)
                        
                        # Calculate similarity
                        sim = torch.cosine_similarity(text_features, image_features, dim=1).item()
                        results.append((img_url, sim, image))
                    except:
                        continue
                
                # Show top results
                results.sort(key=lambda x: x[1], reverse=True)
                cols = st.columns(num_results)
                for i, (url, score, img) in enumerate(results[:num_results]):
                    with cols[i]:
                        st.image(img, use_column_width=True)
                        st.progress(score, text=f"Match: {score:.0%}")

    # Unique Feature 3: Audio Classification with Live Recording
    elif task == "audio_classification":
        st.subheader("🎵 What's That Sound?")
        
        audio_source = st.radio("Audio source", ["Record", "Upload"])
        
        if audio_source == "Record":
            audio_bytes = st.audio("record", format="audio/wav")
        else:
            audio_file = st.file_uploader("Upload audio", type=["wav", "mp3"])
        
        if (audio_source == "Record" and audio_bytes) or (audio_source == "Upload" and audio_file):
            if st.button("Analyze Sound"):
                with st.spinner("Identifying sound..."):
                    # Load audio model
                    processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
                    model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base-960h")
                    
                    # Process audio
                    waveform, sample_rate = torchaudio.load(audio_file if audio_source == "Upload" else audio_bytes)
                    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
                    
                    # Classify
                    with torch.no_grad():
                        logits = model(**inputs).logits
                    predicted_class_ids = torch.argmax(logits, dim=-1).item()
                    predicted_label = model.config.id2label[predicted_class_ids]
                    
                    # Show results
                    st.success(f"Predicted sound: **{predicted_label}**")
                    
                    # Confidence visualization
                    probs = torch.nn.functional.softmax(logits, dim=-1).numpy()[0]
                    top5 = sorted(zip(probs, model.config.id2label.values()), reverse=True)[:5]
                    
                    fig, ax = plt.subplots()
                    ax.barh([label for prob, label in top5], [prob for prob, label in top5])
                    ax.set_xlabel("Confidence")
                    st.pyplot(fig)

    # Unique Feature 4: Model Comparison Arena
    elif task == "model_comparison":
        st.subheader("🤝 Model Face-Off")
        
        col1, col2 = st.columns(2)
        with col1:
            model1 = st.selectbox(
                "Model 1",
                options=["gpt2", "facebook/opt-350m", "EleutherAI/gpt-neo-1.3B"],
                index=0
            )
            prompt1 = st.text_area("Prompt for Model 1", "The future of AI will")
            
        with col2:
            model2 = st.selectbox(
                "Model 2",
                options=["gpt2", "facebook/opt-350m", "EleutherAI/gpt-neo-1.3B"],
                index=1
            )
            prompt2 = st.text_area("Prompt for Model 2", prompt1)
        
        if st.button("Run Comparison"):
            with st.spinner("Running models..."):
                # Run both models
                pipe1 = pipeline("text-generation", model=model1)
                result1 = pipe1(prompt1, max_length=50)[0]["generated_text"]
                
                pipe2 = pipeline("text-generation", model=model2)
                result2 = pipe2(prompt2, max_length=50)[0]["generated_text"]
                
                # Display comparison
                st.subheader("Results")
                tab1, tab2 = st.tabs([f"Model 1: {model1}", f"Model 2: {model2}"])
                
                with tab1:
                    st.markdown(f"```\n{result1}\n```")
                
                with tab2:
                    st.markdown(f"```\n{result2}\n```")
                
                # Add evaluation metrics
                st.subheader("Evaluation")
                col1, col2 = st.columns(2)
                
                with col1:
                    pref = st.radio("Which do you prefer?", ["Model 1", "Model 2", "Tie"], horizontal=True)
                
                with col2:
                    if pref != "Tie":
                        st.write(f"You preferred **{pref}**!")
                        if st.button("Save Preference"):
                            # In a real app, you'd log this to a database
                            st.toast("Preference saved for model improvement!")

    # Unique Feature 5: Embedding Space Visualizer
    elif task == "embedding_visualizer":
        st.subheader("🧠 Visualize Embedding Space")
        
        text_inputs = st.text_area(
            "Enter phrases to visualize (one per line)", 
            "king\nqueen\nman\nwoman\ndog\ncat"
        )
        
        if st.button("Generate Visualization"):
            with st.spinner("Creating embedding visualization..."):
                from sklearn.decomposition import PCA
                import plotly.express as px
                
                # Get embeddings
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
                pipe = pipeline("feature-extraction", model=model_name)
                
                texts = [t.strip() for t in text_inputs.split("\n") if t.strip()]
                embeddings = pipe(texts)
                
                # Reduce dimensions
                pca = PCA(n_components=3)
                reduced = pca.fit_transform(np.array(embeddings).mean(axis=1))
                
                # Create 3D plot
                fig = px.scatter_3d(
                    x=reduced[:,0],
                    y=reduced[:,1],
                    z=reduced[:,2],
                    text=texts,
                    title="3D Projection of Text Embeddings"
                )
                fig.update_traces(marker=dict(size=5))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show similarity matrix
                st.subheader("Similarity Matrix")
                sim_matrix = np.inner(embeddings, embeddings)
                fig2 = px.imshow(
                    sim_matrix,
                    x=texts,
                    y=texts,
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig2, use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("🔍 Try different modes to explore multimodal AI capabilities!")