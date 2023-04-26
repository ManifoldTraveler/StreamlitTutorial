import streamlit as st
import pandas as pd
import numpy as np

st.title('Streamlit :sunglasses:')

'''
:red[_Streamlit_] es un marco de aplicación de código abierto para equipos de aprendizaje automático y ciencia de datos.
La cual nos ayuda a creear aplicaciones web en minutos y de manera facil.
'''


st.header('Instalación ')

'''
:blue[Prerequsitos]

Antes de comenzar para hacer uso de streamlilt necesitamos instalar los siguientes programamas

1. Python 3.7 - Python 3.11
2. PIP
3. Anaconda
4. IDE(VsCode,Spyder,Jupyter, ...)


'''
st.subheader('Creando el entorno Virtual')
'''
Para comenzar a hacer uso de streamlit es necesario crear un entorno virtual en conda,
para ello necesitamos abrir una terminal en Anaconda. Esto nos ayudara a portear  la aplicación facilmente puesto
que dicho entorno contendra unicamente las dependencias necesarias para ejecutar nuestra aplicación.

Una vez en la terminal de anaconda notaremos que al principio de la linea de comandos se encuentra la palabra
:blue[_base_], la cual denota el entorno virtual por defecto. A continuación  escribiremos el siguiente comando:
'''
st.code('conda create -n stenv python=3.9',language='bash')


'''Una vez ejecutado este comando sera necesario esperar a que se instalen los paquetes necesarios
para el correcto funcionamiento del entorno. Posteriormente realizado este proceso necesitaremos
activar el entorno que hemos creado para ello escribiremos el siguiente comando:

'''
st.code('conda activate stenv',language='bash')

'''
Una vez realizado este proceso notaras que la palabra :blue[_base_]  que se encuentra al principio de la linea
de comandos cambia a :red[stenv]. Y listo ya tienes configurado tu entorno virtual para trabajar con streamlit;
recuerda que una vez que cierres la consola el entorno regresara al que esta dado por defecto :blue[_base_] por lo
que unicamente necesitaras volver a introducir el comando de activacion mostrado previamente.
'''

st.subheader('Instalando Streamlit ')

'''
Una vez que hemos creado y activado el entorno virtual podemos comenzar a instalar streamlit para ello en la
consola con el entorno virtual  :red[stenv] activado ejecutaremos el siguiente comando:
'''
st.code('pip install streamlit',language='bash')

'''Y listo ya tienes streamlit instalado :sunglasses:'''


st.subheader('Verificando la instalación')

'''Para verificar que hemos instalado streamlit de manera correcta sera necesario ejecutar el siguiente comando:'''

st.code('streamlit hello',language='bash')

'''Esto desplegara la aplicacion demo de streamlit y con ello verificaremos que streamlit se encuentra funcionando de
manera correcta.'''


st.caption('Recuerda que los modulos de python que utilizes deben de ser instalados de igual forma en el entorno :red[stenv].')
st.code('pip install modulo_a_utilizar',language='bash')
