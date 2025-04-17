import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from utils.config import TRANSFORMER_MODEL
from langchain.utilities import WikipediaAPIWrapper

def show():
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
    model_name = TRANSFORMER_MODEL
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
