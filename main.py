import streamlit as st
import pandas as pd
from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
import requests
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
import pickle
import os
from openai.embeddings_utils import get_embedding
import openai
import json
from utils import detect_entities, similarity_search_threshold, unspecificity_detector, rubro_decisor, unspecificity_explainer

openai.api_key = "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"


db = {}
for set in ['all', 'curated']:
    # Load the FAISS DB
    pass
    db[set] = FAISS.load_local(
        f"indexes/{set}_rubros_index", OpenAIEmbeddings(openai_api_key=openai.api_key))

st.title('Rubro Decider App')

client_text = st.text_input(
    'Enter a client\'s text:', 'Marca de decoración hecha a mano')

if st.button('Submit'):
    entities = detect_entities(client_text)
    rubros = pd.Series()
    for ent in entities:
        rubros = pd.concat([rubros, similarity_search_threshold(
            db['curated'], ent, 0.3, 25)['page_content']])
    if rubros.empty:
        rubros = pd.Series(similarity_search_threshold(
            db['all'], client_text, 0.3, 25)['page_content'])
    rubros = list(rubros.values)

    nl = '\n- '
    st.write(f"El texto provisto por el cliente es: '{client_text}'\n")
    st.write(f"Las entidades detectadas son: {nl}{nl.join(entities)}\n")
    st.write(f"Los rubros detectados son: {nl}{nl.join(rubros)}\n")

    if len(rubros) == 0:
        unspecificity_explainer(client_text)
        st.write(unspecificity_explainer)

    if len(rubros) > 0:
        unespecificResponse = unspecificity_detector(client_text, rubros)
        st.write(unespecificResponse)

    if len(rubros) > 0:
        rubrosResponse = rubro_decisor(client_text, rubros)
        st.write(rubrosResponse)
