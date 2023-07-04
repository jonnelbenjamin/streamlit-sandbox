import streamlit as st
import spacy
from spacy import displacy
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from langchain.utilities import WikipediaAPIWrapper
import wikipedia


nlp = spacy.load('en_core_web_md')


# Display a section header:
st.header("Dependency visualizer")
# st.text_input takes a label and default text string:
input_text = st.text_input("Write some text here to analyze: ", "")
# Send the text string to the SpaCy nlp object for converting to a 'doc' object.
doc = nlp(input_text)

# Use spacy's render() function to generate SVG.
# style="dep" indicates dependencies should be generated.
dep_svg = displacy.render(doc, style="dep", jupyter=False)
st.image(dep_svg, width=400, use_column_width="never")

# Add a section header:
st.header("Entity visualizer")
# Take the text from the input field and render the entity html.
# Note that style="ent" indicates entities.
ent_html = displacy.render(doc, style="ent", jupyter=False)
# Display the entity visualization in the browser:
st.markdown(ent_html, unsafe_allow_html=True)

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