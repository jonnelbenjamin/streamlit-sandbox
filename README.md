# streamlit-sandbox
Sandbox for testing features and use cases for streamlit and huggingface libraries

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


#### Folder Structure
my_app/
├── app.py                  # Main entry point
├── components/
│   ├── __init__.py
│   ├── dependency_viz.py   # Dependency visualizer
│   ├── entity_viz.py       # Entity visualizer
│   ├── qa_system.py       # Question answering
│   ├── image_gen.py       # Image generator
│   └── ner_stats.py       # NER statistics
└── utils/
    ├── __init__.py
    └── config.py          # Shared configurations