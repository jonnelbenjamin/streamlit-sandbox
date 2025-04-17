import streamlit as st
import spacy
from spacy import displacy
from streamlit.components.v1 import html

# Load the latest spaCy model (replace with 'en_core_web_trf' for transformer-based)
nlp = spacy.load("en_core_web_lg")
# Custom Streamlit NER visualizer
def visualize_ner(text, model=nlp):
    doc = model(text)
    html_content = displacy.render(doc, style="ent", page=True)
    return html_content

def show():
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
    nlp = spacy.load("en_core_web_lg")
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
