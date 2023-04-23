import streamlit as st
import pandas as pd
import numpy as np
import sympy as sy
from matplotlib import pyplot as plt
from sympy.plotting.plot import MatplotlibBackend, Plot
from sympy.plotting import plot3d,plot3d_parametric_line
import plotly as ply


def get_sympy_subplots(plot:Plot):
    """
    It takes a plot object and returns a matplotlib figure object

    :param plot: The plot object to be rendered
    :type plot: Plot
    :return: A matplotlib figure object.
    """
    backend = MatplotlibBackend(plot)

    backend.process_series()
    backend.fig.tight_layout()
    return backend.plt

def li(v, i):
    """
    The function takes a list of numbers and an index, and returns the Lagrange interpolating polynomial for the list of
    numbers with the index'th number removed

    :param v: the list of x values
    :param i: the index of the x value you want to interpolate
    :return: the Lagrange interpolating polynomial for the given data points.
    """
    x = sy.symbols('x')

    s = 1
    st = ''
    for k in range(0,len(v)):
        if k != i:
            st = st + '((' + str(x) + '-'+ str(v[k])+')/('+str(v[i])+'-'+str(v[k])+'))'
            s = s*((x-v[k])/(v[i]-v[k]))

    return s

def Lagrange(v,fx):
    """
    It takes in a list of x values and a list of y values, and returns the Lagrange polynomial that interpolates those
    points

    :param v: list of x values
    :param fx: The function you want to interpolate
    :return: the Lagrange polynomial.
    """
    print(v)
    print(fx)
    lis = []
    for i in range(0,len(v)):
        lis.append(li(v,i))

    sums = 0

    for k in range(0,len(v)):
        sums = sums+(fx[k]*lis[k])

    print(sums)

    sy.simplify(sums)

    sy.pprint(sums)

    p1 = sy.plot(sums,show=False)
    p2 = get_sympy_subplots(p1)
    p2.plot(v,fx,"o")
    #p2.show()
    return sy.expand(sums), p2

st.title(':blue[Interpolación de Lagrange]')

st.subheader(':blue[Descripción del método]')

st.subheader(':blue[Ejemplo]')



xxs = st.text_input('Ingrese los valores de x: ',value='{1,2,3,4}')

xsstr = ''


for i in xxs:

    if i != '{' and i != '}' and i != '[' and i != ']' and i != '(' and i != ')' and i != ' ':
        xsstr = xsstr + i

fxxs = st.text_input('Ingrese los valores de f(x): ',value='{-1,3,4,5}')

x = list(map(float,xsstr.split(',')))
intstrr = ''




for t in fxxs:

    if t != '{' and t != '}' and t != '[' and t != ']' and t != '(' and t != ')' and t != ' ':
        intstrr = intstrr + t

fx = list(map(float,intstrr.split(',')))


#st.write(x)
#st.write(fx)



method = Lagrange(x,fx)
st.write('_El polinomio de Interpolacion está dado por:_')
st.latex(sy.latex(method[0]))

st.pyplot(method[1])

