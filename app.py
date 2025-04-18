import streamlit as st
import spacy
from spacy import displacy
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

import wikipedia

import pandas as pd
from components import dependency_viz, entity_viz, qa_system, image_gen, ner_stats, huggingface_multimodal, forecasting, h2o_automl, geopandas_demo

# Page configuration (moved to top so it applies to all pages)
st.set_page_config(page_title="ML Sandbox", page_icon="🤖", layout="wide")

nlp = spacy.load('en_core_web_sm')

# Sidebar navigation
st.sidebar.title("Choose a Module")
app_mode = st.sidebar.radio("", [
    "Welcome",
    "Dependency Visualizer",
    "Entity Visualizer",
    "QA System",
    "Image Generator",
    "NER Statistics",
    "Multimodal Playground",
    "Prophet Forecasting",
    "H2O AutoML",
    "Geopandas"
], index=0)

# Welcome page function
def show_welcome():
    st.title("Welcome to the Machine Learning Sandbox! 🚀")
    
    st.markdown("""
    ## Explore Cutting-Edge ML Models and Libraries
    
    This sandbox provides an interactive environment to experiment with various state-of-the-art 
    machine learning models and libraries. Whether you're interested in natural language processing, 
    computer vision, or forecasting, we've got you covered!
    
    ### What You Can Do Here:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **Analyze text** with dependency parsing and entity recognition
        - **Ask questions** to our question-answering system
        - **Generate images** using AI models
        - **Visualize named entities** in your text
        """)
        
    with col2:
        st.markdown("""
        - **Interact with multimodal models** combining text and images
        - **Create forecasts** with Prophet time series analysis
        - **Explore model capabilities** through intuitive interfaces
        - **Compare results** across different approaches
        """)
    
    st.markdown("""
    ### Getting Started
    
    To begin, select any module from the sidebar on the left. Each module offers unique 
    features and capabilities to help you understand and leverage machine learning technologies.
    
    Happy exploring! 🔍
    """)
    
    # Example visualization to make the welcome page more engaging
    st.subheader("Featured Technologies")
    tech_data = {
        'Technology': ['NLP', 'Computer Vision', 'Q&A Systems', 'Time Series', 'Multimodal AI'],
        'Applications': [85, 78, 70, 65, 60]
    }
    tech_chart = pd.DataFrame(tech_data)
    st.bar_chart(tech_chart.set_index('Technology'))

# Dynamic rendering
if app_mode == "Welcome":
    show_welcome()
elif app_mode == "Dependency Visualizer":
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
elif app_mode == "H2O AutoML":
    h2o_automl.show()
elif app_mode == "Geopandas":
    geopandas_demo.show_geopandas_demo()


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

