import streamlit as st
import pandas as pd
import numpy as np

st.header(":red[Input]")

'''Podemos leer entradas de texto con la funcion st.text_input la cual nos permite almacenar
una cadena de caracteres ingresada por el usuario.'''

n=st.text_input('Ingrese su nombre:')

st.write('Mi nombre es '+n)


'''De forma analoga podemos leer entradas numericas con la funcion st.number_input la cual acepta unicamente entradas
de tipo numericas.'''

num = st.number_input('Ingrese un numero:')

st.write('El numero es ' + str(num))
