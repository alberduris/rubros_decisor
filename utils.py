import openai
import json
import pandas as pd
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from langchain.output_parsers import  OutputFixingParser

# TODO: Integrar el OutputParser al resto de prompts

import streamlit as st

## OTHER FUNCTIONS ##


def similarity_search_threshold(db, query="", threshold=0.3, max=10):
    # [(Document(page_content='[Decoración] Velas', metadata={}), 0.21071842), ... ]
    ssws = db.similarity_search_with_score(query=query, k=max)

    # Create a pandas df from ssws
    df = pd.DataFrame([(doc.page_content, score) for doc, score in ssws], columns=[
        'page_content', 'score'], index=None)

    # Filter by threshold: Get all the rows where the score is less than the threshold and if it's none then return the five first rows
    df_filtered = df[df['score'] < threshold]

    return df_filtered


def parse_output(initial_response, parser, model, max_retries=3):
    try:
        return json.loads(initial_response.choices[0].message.content)
    except:
        print(
            "No JSON object could be decoded, trying OutputFixingParser...")
        # Instantiate the OutputFixingParser
        fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI(
            openai_api_key=openai.api_key, temperature=0, model_name=model))

        # Loop max_retries times while the response is not valid JSON
        to_fix_response = initial_response
        for i in range(max_retries):
            fixed_response = fixing_parser.parse(
                to_fix_response)  # Get the fixed response (class)
            print(f"Retrying {i+1}/{max_retries}...")
            try:
                return fixed_response.dict()
            except:
                # If still not valid JSON, keep trying looping over response
                print("Another one bites the dust")
                to_fix_response = fixed_response

        print(
            "No JSON object could be decoded, returning as string")
        return initial_response.choices[0].message.content


## PROMPTS ##

def detect_entities(text, model):
    system_prompt = """Eres un detector de entidades (Named Entity Recognition system). 

Contexto: Como NER, tu tarea es detectar PRODUCTOS, SERVICIOS Y PROFESIONES en un texto dado. Debes copiar literalmente cada PRODUCTO, SERVICIO Y PROFESIÓN tal y como aparece en el mensaje original.

Instrucciones: Tu respuesta debe ser SIEMPRE una lista de strings en JSON válido siguiendo el SCHEMA.

SCHEMA:
["<entidad1>", "<entidad2>", ..., "<entidadn>"]

Instrucción: Si no hay ninguna entidad, devuelve una lista vacía ([]).

Tarea: Lee el texto, extrae cada PRODUCTO, SERVICIO y PROFESIÓN y escribe una lista en formato JSON válido siguiendo el SCHEMA."""

    # Call to OpenAI completion endpoint with GPT and system_prompt
    response = openai.ChatCompletion.create(model=model, messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ],
        temperature=0,)
    try:
        return json.loads(response.choices[0].message.content)
    except:
        print(
            "detect_entities > No JSON object could be decoded, returning as string")
        return response.choices[0].message.content


class Unspecificity(BaseModel):
    inespecifico: str = Field(
        description="Las posibles respuesta, siendo 'Sí', 'No' o 'Quizás'")
    tipo: str = Field(description="El tipo de inespecificidad")
    fragmento: str = Field(description="El fragmento de TextoCliente")
    justificacion: str = Field(description="La justificación")


def unspecificity_detector(text, rubros, model, max_retries=3):
    system_msg = """Eres un abogado experto en el registro de marcas comerciales. Las marcas se clasifican en base al rubro al cual se dedican. 

A partir de ahora debes actuar como un clasificador de texto multi-clase. 

Recibirás un texto libre identificado por "TextoCliente" y una serie de rubros relacionados identificados como "PosiblesRubros". El string "TextoCliente" es la descripción de las actividades de la marca del cliente. Cada "TextoCliente" puede ajustarse a uno o varios rubros. Vas a actuar como un clasificador de "inespecificidad". La "inespecificidad" consiste en un "TextoCliente" que no permite determinar uno o varios rubros de la lista "PosiblesRubros". Si permite uno o varios y no hay duda sobre otros, entonces no es inespecífico.

Clasificador: Tu tarea consiste en clasificar el "TextoCliente" como "inespecífico" siendo las posibles respuestas (labels): "Sí", "No", "Quizás".

Tipo: En caso de que la respuesta sea "Sí", tendrás que asignar un tipo de inespecificidad según tu criterio: texto libre.

Fragmento: En caso de que la respuesta sea "Sí", tendrás que identificar qué fragmento o fragmentos específicos del "texto cliente" motivan la respuesta. Copia y pega fragmentos de "TextoCliente" tal y como aparecen en el mensaje original. 

Justificación: En caso de que la respuesta sea "Sí", también tendrás que escribir una breve justificación según tu criterio: texto libre.

{format_instructions}

Tarea: Lee el texto, determina si es inespecífico y genera un string en formato JSON válido siguiendo el SCHEMA.
"""

    # Instantiate the parser based on the Python class that defines the schema
    parser = PydanticOutputParser(pydantic_object=Unspecificity)


    # Instantiate the PromptTemplate class with the system prompt (f-string) as template and the instructions from the parser as the "format_instructions" partial variable
    system_prompt = PromptTemplate(
        template=system_msg,
        input_variables=[],
        partial_variables={
            "format_instructions": parser.get_format_instructions()}
    )

    # Get the actual prompt to be sent to the API by calling the format_prompt method
    system_input = system_prompt.format_prompt()


    user_prompt = PromptTemplate(
        template="TextoCliente: {text}\nPosiblesRubros: {rubros}",
        input_variables=["text", "rubros"],
    )

    user_input = user_prompt.format_prompt(text=text, rubros=rubros)

    # Call to OpenAI completion endpoint with GPT and system_prompt
    initial_response = openai.ChatCompletion.create(model=model, messages=[
        {"role": "system", "content": system_input.to_string()},
        {"role": "user", "content": user_input.to_string()}
    ],
        temperature=0,)

    return parse_output(initial_response, parser, model, max_retries)


def rubro_decisor(text, rubros, model):
    system_prompt = """Eres un abogado experto en el registro de marcas comerciales. Las marcas se clasifican en base al rubro (categoría, conjunto de artículos de consumo de un mismo tipo o relacionados con determinada actividad) al cual se dedican. 

A partir de ahora debes actuar como un clasificador de texto multi-etiqueta. 

Recibirás un texto libre identificado por "TextoCliente" y una serie de rubros relacionados identificados como "PosiblesRubros". El string "TextoCliente" es la descripción de las actividades de la marca del cliente. Cada "TextoCliente" puede ajustarse a uno o varios rubros. Por cada rubro, tienes que decidir si se identifica o no en el "TextoCliente".

Clasificador: Tu tarea consiste en etiquetar el "TextoCliente" con los rubros que le corresponden de la lista "PosiblesRubros". Es posible que no corresponda ningún rubro. Además, como abogado experto, debes devoler el razonamiento asociado a la selección o descarte de cada rubro.

"rubro": el rubro que estás evaluando
"razonamiento": <texto libre>
"decision": "Sí" / "No" / "Quizás" (la decisión viene condicionada por tu razonamiento)

SCHEMA:
[
    {"textocliente": <textocliente>, "rubro": <rubro>, "razonamiento": <razonamiento>, "decision": <decision>},
    ...
]

Instrucción PENSAMIENTO: Basa toda tu respuesta en el "TextoCliente".
Instrucción PENSAMIENTO: En el razonamiento evita suposiciones, hipótesis o conjeturas (sugiere, puede, quizás, es posible, tal vez, probablemente, posiblemente, seguramente, podría ser...). Si no estás seguro, la decisión debe ser "No".
Instrucción PENSAMIENTO: Cuando en el razonamiento haya suposiciones, hipótesis o conjeturas (sugiere, puede, quizás, es posible, tal vez, probablemente, posiblemente, seguramente, podría ser...), aplica el pensamiento crítico (observa, analiza, evalúa, infiere, interpreta y sé crítico contigo mismo).
Instrucción PENSAMIENTO: Si el razonamiento se basa en suposiciones, hipótesis o conjeturas (sugiere, puede, quizás, es posible, tal vez, probablemente, posiblemente, seguramente, podría ser...), la decisión debe ser "No".
Instrucción PENSAMIENTO: El error "falso positivo" es más grave que el "falso negativo".

Instrucción RESPUESTA:  Si ningún rubro es adecuado, debes devolver una lista vacía.
Instrucción RESPUESTA: La respuesta será SIEMPRE en formato LISTA JSON válida siguendo SCHEMA."""

    user_prompt = f"""TextoCliente: "{text}"
PosiblesRubros: {rubros}"""

    # Call to OpenAI completion endpoint with GPT and system_prompt
    response = openai.ChatCompletion.create(model=model, messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
        temperature=0)
    try:
        return json.loads(response.choices[0].message.content)

    except:
        print("rubro_decisor > No JSON object could be decoded, returning as string")
        return response.choices[0].message.content


def unspecificity_explainer(text, model):
    # Not implemented yet
    system_prompt = """Eres un abogado experto en el registro de marcas comerciales. Las marcas se clasifican en base al rubro al cual se dedican. 

A partir de ahora debes actuar como un abogado. 

Recibirás un texto libre identificado por "TextoCliente". El string "TextoCliente" debería ser la descripción de las actividades de la marca del cliente. Sin embargo, el "TextoCliente" que vas a recibir es INESPECÍFICO. Es decir, el "TextoCliente" debería poder ajustarse a uno o varios rubros, pero en este caso, no hay información suficiente para hacerlo. Tienes que JUSTIFICAR por qué el "TextoCliente" es INESPECÍFICO tal y como lo haría un abogado experto en el registro de marcas comerciales.

Tarea: Escribe la JUSTIFICACIÓN de POR QUÉ es INESPECÍFICO el "TextoCliente". Piensa en todas las posibles opciones y justifica tu respuesta con argumentos legales.

Algunos de los TIPOS de inespecificidad que puedes encontrar son: "Descripción muy amplia" (el texto es muy general), "Incorrecto" (especifica algo incorrecto como con qué esta hecho o cómo esta hecho), agrega algo no determinado (agrega al final "y más", "etc.", "otros", ...), "Vacío" (el mensaje no dice nada), "Inclasificado" (el mensaje habla de un producto o servicio para el que no existe un rubro). Además de estas, puede haber otras.

Instrucción: La respuesta será SIEMPRE un string en formato JSON de acuerdo a SCHEMA.

SCHEMA:
{inespecifico: "Sí",
tipo: "<describe el tipo de inespecificidad>",
fragmento: "<fragmento o fragmentos del TextoCliente que motivan la inespecificidad>",
justificacion: "<tu justificación>"}

Instrucción: La respuesta será SIEMPRE en formato JSON válido siguendo SCHEMA."""

    # Call to OpenAI completion endpoint with GPT and system_prompt
    response = openai.ChatCompletion.create(model=model, messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ],
        temperature=0)
    try:
        return json.loads(response.choices[0].message.content)
    except:
        print(
            "unspecificity_explainer > No JSON object could be decoded, returning as string")
        return response.choices[0].message.content
