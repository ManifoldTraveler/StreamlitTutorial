import streamlit as st
import pandas as pd
import numpy as np
from datetime import time, datetime


st.header(':red[Slider]')


"""
Los sliders (deslizadores) nos sirven para poder establecer un modo
sencillo para dar valores dentro de un rango definido o rangos de valores
para poder utilizarlos. Para esto hacemos uso de st.slider este toma como
argumentos la etiqueta del slider, el valor minimo, el valor maximo y el valor
por default, estos valores pueden ser enteros o flotantes (todos los tipos de datos
deben de ser iguales). Ademas podemos hacer uso del parametro step el cual nos permite
establecer como cambian los valores entre cada paso.
"""

#Ejemplo 1 (Slider para un solo valor)
age = st.slider('Cual es tu edad?', 0, 130, 25,step=10)
st.write("Tengo ", age, 'años')

#ejemplo 2 (Slider de rango)

"""
Podemos establecer sliders para rangos tomando de igual forma los valores de la etiqueta, el valor
minimo, maximo y el valor por default en este caso sera una tupla con el valor minimo del rango y
el valor maximo.
"""

ran = st.slider('Selecciona un rango: ',0.0,10.0,(3.1,5.5))
st.write("El rango es: ", ran)


c =  r'''
#Ejemplo 1 (Slider para un solo valor)
age = st.slider('Cual es tu edad?', 0, 130, 25,step=10)
st.write("Tengo ", age, 'años')

#ejemplo 2 (Slider de rango)
ran = st.slider('Selecciona un rango: ',0.0,10.0,(3.1,5.5))
st.write("El rango es: ", ran)

'''

st.code(c,language='python')
