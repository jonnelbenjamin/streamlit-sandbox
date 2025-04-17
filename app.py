import streamlit as st
import spacy
from spacy import displacy
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from langchain.utilities import WikipediaAPIWrapper
import wikipedia
from diffusers import StableDiffusionPipeline
import torch
import pandas as pd
from components import dependency_viz, entity_viz


nlp = spacy.load('en_core_web_sm')

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a module", [
    "Dependency Visualizer",
    "Entity Visualizer",
    "QA System",
    "Image Generator",
    "NER Statistics"
])

# Dynamic rendering
if app_mode == "Dependency Visualizer":
    dependency_viz.show()
elif app_mode == "Entity Visualizer":
    entity_viz.show()


# Footer (shared across all pages)
st.sidebar.markdown("---")
st.sidebar.markdown("### Created by Jonnel Benjamin")

# Set favicon
# st.set_page_config(page_title="Streamlit App", page_icon="static/res/favicon.png")

st.markdown(
    '''
    <style>
        .center-image {
            display: flex;
            justify-content: center;
        }
    </style>
 
    <body>
        <header>
            <div>
                <h1>Streamlit Question Answering App</h1>
                <div class="center-image">
                <h1>🦜 🦚</h1>
                </div>
            </div>
        </header>
    </body>
    ''',
    unsafe_allow_html=True
)

# Load the question answering model and tokenizer
model_name = "deepset/roberta-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a pipeline for question answering
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

# User input
question_input = st.text_input("Question:")

if question_input:
    # Extract keywords from the question input
    keywords = question_input.split()

    # Fetch context information using the Wikipedia toolkit based on keywords
    wikipedia = WikipediaAPIWrapper()
    context_input = wikipedia.run(' '.join(keywords))

    # Prepare the question and context for question answering
    QA_input = {
        'question': question_input,
        'context': context_input
    }

    # Get the answer using the question answering pipeline
    res = nlp(QA_input)

    # Display the answer
    st.text_area("Answer:", res['answer'])
    st.write("Score:", res['score'])


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

import spacy
from spacy import displacy
import streamlit as st
from streamlit.components.v1 import html

# Load the latest spaCy model (replace with 'en_core_web_trf' for transformer-based)
nlp = spacy.load("en_core_web_lg")

# Custom Streamlit NER visualizer
def visualize_ner(text, model=nlp):
    doc = model(text)
    html_content = displacy.render(doc, style="ent", page=True)
    return html_content

# Streamlit UI with modern features
st.title("🔍Named Entity Recognition with spaCy & Streamlit")

# Text input with a cool placeholder
user_input = st.text_area(
    "Enter text to analyze:",
    "Apple is looking to buy a U.K. startup for $1 billion in 2024.",
    height=150,
)

# Add a toggle for transformer model (cutting-edge)
use_transformer = st.toggle("🚀 Use Transformer Model (en_core_web_trf)", False)
if use_transformer:
    nlp = spacy.load("en_core_web_trf")

# Custom entity colors (spaCy 3.5+)
colors = {"ORG": "#FF5733", "GPE": "#33FF57", "DATE": "#3357FF"}
nlp.get_pipe("ner").add_label("ORG")

# Process and display
if st.button("Analyze Text"):
    with st.spinner("🔍 Detecting entities..."):
        ner_html = visualize_ner(user_input)
        html(ner_html, height=300, scrolling=True)

    # Display raw JSON (for debugging)
    with st.expander("📦 See raw spaCy doc"):
        doc = nlp(user_input)
        st.json(doc.to_json())

# Bonus: Entity frequency bar chart
# Had to use this way to tilt x-axis titles 45 degrees
# Switch to Matplotlib for more granular control
if st.button("Show Entity Stats"):
    doc = nlp(user_input)
    entities = [ent.label_ for ent in doc.ents]
    if entities:
        # Convert to DataFrame for better formatting
        df = pd.Series(entities).value_counts().reset_index()
        df.columns = ['Entity', 'Count']
        
        # Use altair for more customization
        import altair as alt
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('Entity:O', axis=alt.Axis(labelAngle=-45)),
            y='Count'
        )
        st.altair_chart(chart, use_container_width=True)

st.markdown(
    '''
    <style>
        .center-image {
            display: flex;
            justify-content: center;
        }
        .follow-me {
            text-align: center;
        }
        .social-icons {
            display: flex;
            justify-content: center;
            list-style: none;
            padding: 0;
        }
        .social-icons li {
            margin: 0 10px;
        }
    </style>
    <body>
        <div class="center-image">
            <h4>Jonnel Benjamin 🤖</h4>
        </div>
       
    </body>
    ''',
    unsafe_allow_html=True
)