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


st.sidebar.header('API Key Configuration')
with st.sidebar:
    api_key = st.text_input('OpenAI API key')
    model = st.selectbox('Model', ['gpt-3.5-turbo', 'gpt-4'])


st.title('Parte 1')
st.header('Detector de inespecificidades y rubros')


if api_key == '':
    st.warning('No se ha configurado una API key de OpenAI')

else:
    openai.api_key = api_key
    db = {}
    for set in ['all', 'curated']:
        # Load the FAISS DB
        pass
        db[set] = FAISS.load_local(
            f"indexes/{set}_rubros_index", OpenAIEmbeddings(openai_api_key=openai.api_key))

    client_text = st.text_input(
        'Texto del cliente:', '', placeholder='Marca de decoración hecha a mano')

    if st.button('Submit'):
        with st.spinner('Detectando entidades y extrayendo rubros relacionados...'):
            entities = detect_entities(client_text, model)
            entities = "lalalala nononono"
            rubros = pd.Series(dtype='object')
            # Check if entities is a list
            if not isinstance(entities, list) or len(entities) == 0:
                entities = [client_text]
            for ent in entities:
                rubros = pd.concat([rubros, similarity_search_threshold(
                    db['curated'], ent, 0.3, 25)['page_content']])
            if rubros.empty:
                for ent in entities:
                    rubros = pd.Series(similarity_search_threshold(
                        db['all'], ent, 0.3, 25)['page_content'])

            rubros = list(rubros.values)

        col1, col2 = st.columns(2)
        nl = '\n- '
        with col1:
            st.markdown(f"**Las entidades detectadas son:**")
            with st.expander("Ver entidades"):
                st.write(f"{nl}{nl.join(entities)}\n")
                st.write("\n")
        with col2:
            # Check if rubros is empty
            st.markdown(f"**Los posibles rublos relacionados son:**")
            if len(rubros) == 0:
                st.warning("No se encontraron rubros relacionados")
            else:
                with st.expander("Ver rubros"):
                    st.write(f"{nl}{nl.join(rubros)}\n")
                    st.write("\n")

        if len(rubros) > 0:
            with st.spinner('Detectando inespecificidades...'):
                unespecificResponse = unspecificity_detector(
                    client_text, rubros, model)

            st.markdown(f"**Inespecificidad:**")
            try:
                st.json(unespecificResponse)
            except:
                st.write(unespecificResponse)

            with st.spinner("Evaluando rubros..."):
                rubrosResponse = rubro_decisor(client_text, rubros, model)

            st.markdown(f"**Rubros:**")

            # If rubrosResponse is JSON object
            if isinstance(rubrosResponse, list):
                accepted_rubros = list(
                    filter(lambda x: x["decision"] == 'Sí', rubrosResponse))
                maybe_rubros = list(
                    filter(lambda x: x["decision"] == 'Quizás', rubrosResponse))
                rejected_rubros = list(
                    filter(lambda x: x["decision"] == 'No', rubrosResponse))

                if len(accepted_rubros) == 0:
                    st.warning("No se encontraron rubros aceptados")
                    # Show rejected rubros
                    if len(rejected_rubros) > 0:
                        st.markdown("**Rubros rechazados**")
                        for rubro in rejected_rubros:
                            with st.expander(rubro["rubro"]):
                                st.error(rubro["razonamiento"])
                else:
                    st.markdown("**Rubros aceptados**")
                    for rubro in accepted_rubros:
                        with st.expander(rubro["rubro"]):
                            st.info(rubro["razonamiento"])
                if len(maybe_rubros) > 0:
                    st.markdown("**Rubros en duda**")
                    for rubro in maybe_rubros:
                        with st.expander(rubro["rubro"]):
                            st.warning(rubro["razonamiento"].replace(
                                "TextoCliente", "texto del cliente"))
            else:
                try:
                    print(rubrosResponse)
                    st.json(rubrosResponse)
                except:
                    st.write(rubrosResponse)

        if len(rubros) == 0:
            with st.spinner("Extrayendo razones de inespecificidad..."):
                unspecificExplanation = unspecificity_explainer(
                    client_text, model)
                st.markdown(f"**Inespecificidad:**")
                try:
                    st.json(unspecificExplanation)
                except:
                    st.write(unspecificExplanation)
