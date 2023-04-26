import streamlit as st
import pandas as pd
import numpy as np

st.header(":red[Select Box]")


"""
Para poder dar una lista de opciones podemos hacer uso de st.selectbox la cual
nos ayudara a poder dar una lista de valores para elegir. Para ello pasamos como
argumentos la etiqueta(string), una tupla con las opciones a elegir.
"""

sb = st.selectbox('Cual es tu color favorito?', ('rojo','azul','verde'))
c = ''

if sb == 'rojo':
    c = 'red'
elif sb == 'azul':
    c = 'blue'
elif sb == 'verde':
    c = 'green'





st.write("Tu color favorito es:  :"+c+"["+str(sb)+"]")



c = r'''
sb = st.selectbox('Cual es tu color favorito?', ('rojo','azul','verde'))
c = ''

if sb == 'rojo':
    c = 'red'
elif sb == 'azul':
    c = 'blue'
elif sb == 'verde':
    c = 'green'





st.write("Tu color favorito es:  :"+c+"["+str(sb)+"]")

'''
st.write('Codigo Utilizado:')
st.code(c,language='python')
