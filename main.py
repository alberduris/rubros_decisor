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
import warnings
from utils import detect_entities, similarity_search_threshold, unspecificity_detector, rubro_decisor, unspecificity_explainer


def is_json(myjson):
    # If myJson is a dict return true
    if isinstance(myjson, dict):
        return True
    try:
        json.loads(myjson)
    except ValueError as e:
        return False
    return True


with st.sidebar:
    st.header('API Key Configuration')
    api_key = st.text_input('OpenAI API key')
    model = st.selectbox('Model', ['gpt-3.5-turbo', 'gpt-4'])

    st.header('Búsqueda semántica')
    ss_type = st.selectbox('Selecciona la lista de rubros a utilizar',
                           ['Automático', 'Lista curada', 'Lista completa'])
    if ss_type == 'Automático':
        st.success(
            'Se emplea la lista curada de rubros por defecto y la lista completa como fallback en caso de no encontrar coincidencias')
    else:
        st.warning(
            f'Se empleará únicamente la lista seleccionada: ({ss_type}).')

    # Slider for selecting the threshold between [0.1, 0.7]
    threshold = st.slider('Umbral de similaridad', 0.1, 0.7, 0.3, 0.01)


st.title('Parte 1')
st.header('Detector de inespecificidades y rubros')


if api_key == '':
    st.warning('No se ha configurado una API key de OpenAI')

else:
    openai.api_key = api_key
    db = {}
    for set in ['all', 'curated']:
        # Load the FAISS DB
        db[set] = FAISS.load_local(
            f"indexes/{set}_rubros_index", OpenAIEmbeddings(openai_api_key=openai.api_key))

    client_text = st.text_input(
        'Texto del cliente:', '', placeholder='Marca de decoración hecha a mano')

    if st.button('Enviar'):
        print('Texto del cliente:', client_text)
        st.markdown("### Extracción de entidades y búsqueda semántica")
        with st.spinner('Detectando entidades y extrayendo rubros relacionados...'):
            entities = detect_entities(client_text, model)
            # Add the client text to entities in the first position of the list

            searched_rubros = pd.DataFrame()
            # Check if entities is a list
            if not isinstance(entities, list) or len(entities) == 0:
                entities = [client_text]
            else:
                entities.insert(0, client_text)

            # Semantic search for rubros
            if ss_type == 'Lista curada' or ss_type == 'Automático':
                if ss_type == 'Lista curada':
                    st.warning(
                        'Se emplea únicamente la lista curada de rubros')
                for ent in entities:
                    searched_rubros = pd.concat([searched_rubros, similarity_search_threshold(
                        db['curated'], ent, threshold=threshold, max=10)])
                if ss_type == 'Automático' and len(searched_rubros) < 5:
                    st.warning(
                        f'Se encontraron {len(searched_rubros)} rubros relacionados en la lista curada, empleando la lista completa de rubros')
                    for ent in entities:
                        searched_rubros = pd.concat([searched_rubros, similarity_search_threshold(
                            db['all'], ent, threshold=threshold, max=10)])
            elif ss_type == 'Lista completa':
                st.warning(
                    'Se emplea únicamente la lista completa de rubros')
                for ent in entities:
                    searched_rubros = pd.concat([searched_rubros, similarity_search_threshold(
                        db['all'], ent, threshold=threshold, max=10)])
                    # st.write(ent)
                    # st.table(searched_rubros.sort_values(by='score'))

            # Sort rubros by score, remove duplicates keeping the one with the lowest score, and get first 10 or entities length as list
            searched_rubros = searched_rubros.sort_values(by='score')
            searched_rubros = searched_rubros.drop_duplicates(
                subset='page_content', keep='first', inplace=False)
            rubros = searched_rubros.head(10 if len(entities) <= 10 else len(entities))[
                "page_content"].tolist()

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
                st.warning(
                    "No se encontraron rubros relacionados. Puedes probar a aumentar el umbral de similaridad.")
            else:
                with st.expander("Ver rubros"):
                    # Rename columns of searched_rubros to "rubro" and "similaridad"
                    st.table(searched_rubros.head(10 if len(entities) <= 10 else len(entities)).rename(inplace=False, columns={
                             "page_content": "rubro", "score": "similaridad"}).assign(similaridad=lambda x: x['similaridad'].apply(lambda y: f"{y:.2f}")).reset_index(drop=True))

        if len(rubros) > 0:
            with st.spinner('Detectando inespecificidades...'):
                unespecificResponse = unspecificity_detector(
                    client_text, rubros, model)

            st.markdown("### Inespecificidad")
            # If unespecificResponse is JSON object
            if is_json(unespecificResponse):
                st.json(unespecificResponse)
            else:
                warnings.warn(
                    "No se pudo parsear la respuesta de inespecificidad")
                st.write(unespecificResponse)

            st.markdown(f"### Rubros")
            with st.spinner("Evaluando rubros..."):
                rubrosResponse = rubro_decisor(client_text, rubros, model)

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
                            st.warning(rubro["razonamiento"])
            else:
                # Try to print JSON object
                if is_json(rubrosResponse):
                    st.json(rubrosResponse)
                else:
                    warnings.warn(
                        "No se pudo parsear la respuesta de los rubros")
                    st.write(rubrosResponse)

        if len(rubros) == 0:
            with st.spinner("Extrayendo razones de inespecificidad..."):
                unspecificExplanation = unspecificity_explainer(
                    client_text, model)
                st.markdown("### Inespecificidad")
                # Try to print JSON object
                if is_json(unspecificExplanation):
                    st.write(unspecificExplanation)
                else:
                    st.write(unspecificExplanation)
