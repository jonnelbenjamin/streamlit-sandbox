import streamlit as st
import spacy
from spacy import displacy
from utils.config import SPACY_MODEL, ENTITY_COLORS

nlp = spacy.load('en_core_web_sm')

def show():
    st.header("📊 Advanced Entity Visualizer")
    
    # Model selection in sidebar
    with st.sidebar.expander("Model Settings"):
        use_transformer = st.checkbox("Use Transformer Model", False)
        model = SPACY_TRF_MODEL if use_transformer else SPACY_MODEL
    
    # Load model with caching
    @st.cache_resource
    def load_model(model_name):
        return spacy.load(model_name)
    
    nlp = load_model(model)
    
    # Text input with example
    input_text = st.text_area(
        "Enter text to analyze:",
        "Apple is looking to buy a U.K. startup for $1 billion in 2024.",
        height=150
    )
    
    if st.button("Analyze Entities"):
        if not input_text.strip():
            st.warning("Please enter some text to analyze")
            return
            
        with st.spinner("Processing text..."):
            doc = nlp(input_text)
            
            # Visualization options
            with st.expander("Display Options"):
                colors = {k: st.color_picker(k, v) 
                         for k, v in ENTITY_COLORS.items()}
                manual = st.checkbox("Manual entity selection")
                
                if manual:
                    selected_ents = st.multiselect(
                        "Entities to display",
                        options=list(ENTITY_COLORS.keys()),
                        default=list(ENTITY_COLORS.keys())
                    )
                    options = {"ents": selected_ents, "colors": colors}
                else:
                    options = {"colors": colors}
            
            # Render visualization
            ent_html = displacy.render(
                doc,
                style="ent",
                options=options,
                jupyter=False
            )
            
            # Display results in two columns
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.subheader("Visualization")
                st.markdown(ent_html, unsafe_allow_html=True)
            
            with col2:
                st.subheader("Entity Details")
                if doc.ents:
                    for ent in doc.ents:
                        st.write(f"🔹 {ent.text} ({ent.label_})")
                else:
                    st.info("No entities found")
