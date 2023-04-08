import openai
import json
import pandas as pd

## OTHER FUNCTIONS ##


def similarity_search_threshold(db, query, threshold, max):
    # [(Document(page_content='[Decoración] Velas', metadata={}), 0.21071842), ... ]
    ssws = db.similarity_search_with_score(query=query, k=max)
    # Create a pandas df from ssws
    df = pd.DataFrame([(doc.page_content, score)
                      for doc, score in ssws], columns=['page_content', 'score'])
    # Filter by threshold: Get all the rows where the score is less than the threshold and if it's none then return the five first rows
    return df.head(5) if len(df[df['score'] < threshold]) < 5 else df[df['score'] < threshold]


## PROMPTS ##

def detect_entities(text):
    system_prompt = """Eres un detector de entidades. 

Contexto: Tu tarea es detectar PRODUCTOS, SERVICIOS Y PROFESIONES en un texto dado. Debes copiar literalmente cada PRODUCTO, SERVICIO Y PROFESIÓN tal y como aparece en el mensaje original.

Instrucciones: Tu respuesta debe ser una lista de strings tal que "["<PRODUCTO1>", "<SERVICIO1>", ..., "<PRODUCTON>"].

Tarea: Lee el texto, extrae cada PRODUCTO, SERVICIO y PROFESIÓN y crea una lista en formato JSON."""
    # Call to OpenAI completion endpoint with GPT and system_prompt
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ],
        temperature=0,)
    return json.loads(response.choices[0].message.content)


def unspecificity_detector(text, rubros):
    system_prompt = """Eres un abogado experto en el registro de marcas comerciales. Las marcas se clasifican en base al rubro al cual se dedican. 

A partir de ahora debes actuar como un clasificador de texto multi-clase. 

Recibirás un texto libre identificado por "TextoCliente" y una serie de rubros relacionados identificados como "PosiblesRubros". El string "TextoCliente" es la descripción de las actividades de la marca del cliente. Cada "TextoCliente" puede ajustarse a uno o varios rubros. Vas a actuar como un clasificador de "inespecificidad". La "inespecificidad" consiste en un "TextoCliente" que no permite determinar uno o varios rubros de la lista "PosiblesRubros". Si permite uno o varios y no hay duda sobre otros, entonces no es inespecífico.

Clasificador: Tu tarea consiste en clasificar el "TextoCliente" como "inespecífico" siendo las posibles respuestas (labels): "Sí", "No", "Quizás".

Tipo: En caso de que la respuesta sea "Sí", tendrás que asignar un tipo de inespecificidad según tu criterio: texto libre.

Fragmento: En caso de que la respuesta sea "Sí", tendrás que identificar qué fragmento o fragmentos específicos del "texto cliente" motivan la respuesta. Copia y pega fragmentos de "TextoCliente" tal y como aparecen en el mensaje original. 

Justificación: En caso de que la respuesta sea "Sí", también tendrás que escribir una breve justificación según tu criterio: texto libre.

La respuesta será SIEMPRE un string en formato JSON con la siguiente estructura:

SCHEMA:
{inespecifico: "Sí" / "No" / "Quizás",
tipo: "<tipo>",
fragmento: "<fragmento de texto cliente>",
justificacion: "<tu justificación>"}

EJEMPLO: 

{inespecifico: "Sí",
tipo: "No se especifica el rubro",
fragmento: "Venta de productos de limpieza",
justificacion: "No se especifica el rubro al cual se dedica la marca"}

{inespecifico: "No",
tipo: "",
fragmento: "",
justificacion: ""}

La respuesta será SIEMPRE en formato JSON válido siguendo SCHEMA.
"""

    user_prompt = f"""TextoCliente: "{text}"
PosiblesRubros: {rubros}"""

    # Call to OpenAI completion endpoint with GPT and system_prompt
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
        temperature=0,)
    # TODO: Add checking for valid JSON and if it's not valid then return an empty response
    return json.loads(response.choices[0].message.content)


def rubro_decisor(text, rubros):
    system_prompt = """Eres un abogado experto en el registro de marcas comerciales. Las marcas se clasifican en base al rubro (categoría, conjunto de artículos de consumo de un mismo tipo o relacionados con determinada actividad) al cual se dedican. 

A partir de ahora debes actuar como un clasificador de texto multi-etiqueta. 

Recibirás un texto libre identificado por "TextoCliente" y una serie de rubros relacionados identificados como "PosiblesRubros". El string "TextoCliente" es la descripción de las actividades de la marca del cliente. Cada "TextoCliente" puede ajustarse a uno o varios rubros. Por cada rubro, tienes que decidir si se identifica o no en el "TextoCliente".

Clasificador: Tu tarea consiste en etiquetar el "TextoCliente" con los rubros que le corresponden de la lista "PosiblesRubros". Es posible que no corresponda ningún rubro. Además, como abogado experto, debes devoler el razonamiento asociado a la selección o descarte de cada rubro.

SCHEMA:
{
    ["textocliente": "<textocliente>", "rubro": "<rubro>", "razonamiento": "<razonamiento>", "decision": "<decision>"],
    ...
}

Instrucción PENSAMIENTO: Basa toda tu respuesta en el "TextoCliente".
Instrucción PENSAMIENTO: Cuando en el razonamiento haya suposiciones, hipótesis o conjeturas (sugiere, puede, quizás, es posible, tal vez, probablemente, posiblemente, seguramente, podría ser...), debes justificar tu respuesta.
Instrucción PENSAMIENTO: Si el razonamiento se basa en suposiciones, hipótesis o conjeturas (sugiere, puede, quizás, es posible, tal vez, probablemente, posiblemente, seguramente, podría ser...), la decisión debe ser "No".
Instrucción PENSAMIENTO: El error "falso positivo" es más grave que el "falso negativo".

Instrucción RESPUESTA:  Si ningún rubro es adecuado, debes devolver una lista vacía.
Instrucción RESPUESTA: Responde con formato JSON válido siguiendo SCHEMA."""

    user_prompt = f"""TextoCliente: "{text}"
PosiblesRubros: {rubros}"""

    # Call to OpenAI completion endpoint with GPT and system_prompt
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
        temperature=0)
    # TODO: Add checking for valid JSON and if it's not valid then return an empty response
    return response.choices[0].message.content


def unspecificity_explainer(text):
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
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ],
        temperature=0)
    # TODO: Add checking for valid JSON and if it's not valid then return an empty response
    return json.loads(response.choices[0].message.content)
