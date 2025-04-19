
# Streamlit Sandbox Components

Explore a collection of interactive ML & Data Science Libraries. Here are the currently available modules:

## 🧩 Available Components

| Component | Description | Key Technologies |
|-----------|------------|------------------|
| **🧠 Dependency Visualizer** | Visualize linguistic dependencies in text | SpaCy, NetworkX |
| **🔍 Entity Visualizer** | Interactive named entity recognition visualization | SpaCy, Streamlit |
| **❓ QA System** | Question answering system with context | Transformers, Haystack |
| **🎨 Image Generator** | Generate images from text prompts | Stable Diffusion, Hugging Face |
| **📊 NER Statistics** | Named Entity Recognition analytics dashboard | SpaCy, Pandas |
| **🖼️ Multimodal Playground** | Experiment with text+image models | CLIP, OpenAI |
| **🔮 Prophet Forecasting** | Time series forecasting tool | Facebook Prophet, Plotly |
| **🤖 H2O AutoML** | Automated machine learning interface | H2O.ai, Scikit-learn |
| **🌍 Geopandas** | Geospatial data analysis and visualization | Geopandas, Folium |
| **📊 Clustering Demo** | Interactive clustering algorithm explorer | Scikit-learn, Plotly |

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/streamlit-sandbox.git
   cd streamlit-sandbox

#### Run the following command for starting application

Create virtual environment:

 `python3 -m venv env`
 
 Activate Virtual environment:
 
 `source env/bin/activate`
 
 To deactivate the virtual environment, run:
 `deactivate`
 
Install necessary libraries:

 - `pip install streamlit`

 - `pip install spacy`

 - `python -m spacy download en_core_web_sm`

 - `python -m spacy download en_core_web_lg`

 - `python -m spacy download en_core_web_trf`

Run the application with the following command:

 `streamlit run app.py`

 ### Make Commands:

- Run `make setup` to install requirements and language models
- Run `make run` to start your Streamlit app
- Use `make clean` when you want to clean up cache files
- Use `make` to execute the default `all` target, which will set up dependencies and then run the app


#### Folder Structure
components/
├── dependency_visualizer.py
├── entity_visualizer.py
├── qa_system.py
├── image_generator.py
├── ner_statistics.py
├── multimodal_playground.py
├── prophet_forecasting.py
├── h2o_automl.py
├── geopandas_demo.py
└── clustering_demo.py
