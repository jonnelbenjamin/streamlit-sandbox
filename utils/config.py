import os
# Shared NLP models
SPACY_MODEL = "en_core_web_lg"
TRANSFORMER_MODEL = "deepset/roberta-base-squad2"
IMAGE_MODEL = "prompthero/openjourney"

# Color schemes
ENTITY_COLORS = {
    "ORG": "#FF5733",
    "GPE": "#33FF57", 
    "DATE": "#3357FF"
}

HF_API_KEY= os.environ["HF_API_KEY"]