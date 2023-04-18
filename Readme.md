# Guía de despliegue

Esta guía explica cómo desplegar la aplicación de Streamlit `rubros_decisor` en Streamlit Community Cloud.

Documentación oficial: https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app.

## Guía paso a paso

### Fork y cuenta en Streamlit Community Cloud

1. Realizar un fork de este repositorio (`rubros_decisor` ) para tener una copia en tu cuenta de GitHub.
2. Crear una cuenta en Streamlit Community Cloud: https://share.streamlit.io/

### Desplegar la aplicación

1. Ve al [Dashboard](https://share.streamlit.io/)
2. Haz click en el botón "New App" en la esquina superior derecha, 
3. En la nueva ventana, elige tu repositorio (`usuario/rubros_decisor`), la branch (`master`) y la ruta de archivo principal (`main.py`), y haz click en "Deploy". 

![deploy](https://github.com/alberduris/rubros_decisor/blob/master/imgs/Screenshot%20from%202023-04-18%2014-11-21.png?raw=true)

Tras eso, la aplicación se desplegará automáticamente y estará disponible en una URL similar a:
  * https://{{usuario}}-rubros-decisor-main-4qzfpr.streamlit.app/

