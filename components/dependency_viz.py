import streamlit as st
import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_sm')

def show():
    # Display a section header:
    st.header("Dependency visualizer")
    # st.text_input takes a label and default text string:
    input_text = st.text_input("Write some text here to analyze: ", "")
    # Send the text string to the SpaCy nlp object for converting to a 'doc' object.
    doc = nlp(input_text)

    # Use spacy's render() function to generate SVG.
    # style="dep" indicates dependencies should be generated.
    dep_svg = displacy.render(doc, style="dep", jupyter=False)
    st.image(dep_svg, width=400, use_container_width="never")
