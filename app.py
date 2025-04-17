import streamlit as st
import spacy
from spacy import displacy
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

import wikipedia

import pandas as pd
from components import dependency_viz, entity_viz, qa_system, image_gen, ner_stats, huggingface_multimodal, forecasting


nlp = spacy.load('en_core_web_sm')

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a module", [
    "Dependency Visualizer",
    "Entity Visualizer",
    "QA System",
    "Image Generator",
    "NER Statistics",
    "Multimodal Playground",
    "Prophet Forecasting"
])

# Dynamic rendering
if app_mode == "Dependency Visualizer":
    dependency_viz.show()
elif app_mode == "Entity Visualizer":
    entity_viz.show()
elif app_mode == "QA System":
    qa_system.show()
elif app_mode == "Image Generator":
    image_gen.show()
elif app_mode == "NER Statistics":
    ner_stats.show()
elif app_mode == "Multimodal Playground":
    huggingface_multimodal.show()
elif app_mode == "Prophet Forecasting":
    forecasting.run_forecasting()


# Footer (shared across all pages)
st.sidebar.markdown("---")

st.sidebar.markdown(
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

# Set favicon
# st.set_page_config(page_title="Streamlit App", page_icon="static/res/favicon.png")

