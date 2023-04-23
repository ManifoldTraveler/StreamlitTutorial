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


def diff_div(v,fx,order):
    """
    > The function takes in a list of values, a list of function values, and an order, and returns a list of divided
    differences

    :param v: the list of x values
    :param fx: the function you want to differentiate
    :param order: the order of the derivative you want to take
    :return: the difference quotient of the function f(x)
    """

    m = []

    for i in range(0,len(fx)):
        #print(fx[i])
        if i + 1 < len(fx) and i +order < len(v):
            #print(v[i+1],v[i],"/",fx[i+order]," ",fx[i])
            m.append((fx[i+1]-fx[i])/(v[i+order]-v[i]))
    return m

def divided_diff(fx,v):
    """
    The function takes in a list of x values and a list of f(x) values, and returns a list of lists of divided differences

    :param fx: the function to be interpolated
    :param v: list of x values
    :return: The divided difference table is being returned.
    """
    x = v
    nfx = fx
    m = []
    for i in range(0,len(v)-1):
        nx = diff_div(v,nfx,i+1)
        #print(nx)
        m.append(nx)
        nfx = nx

    #print(m)
    return m

def Newton_interpolation(fx,v):
    """
    It takes in a list of x values and a list of f(x) values, and returns a polynomial that interpolates the points

    :param fx: a list of the function values
    :param v: list of x values
    :return: The function is being returned.
    """
    diff = divided_diff(fx,v)
    x = sy.symbols('x')

    expr = v[0]

    for i in range(0,len(diff)):
        s = diff[i][0]
        p = 1
        for k in range(0,len(v)):

            p = p*(x-v[k])
            #print(p, "p",k)
            if k == i:
                break
        s = s * p
        expr = expr + s

    #pprint(expr)

    p = sy.plot(expr,(x,-10,10),show=False)
    p2 = get_sympy_subplots(p)
    p2.plot(v,fx,"o")
    #p2.show()

    return sy.expand(expr),p2





st.title(':blue[Interpolación de Newton]')

st.subheader(':blue[Descripción del método]')

st.subheader(':blue[Ejemplo]')



xxs = st.text_input('Ingrese los valores de x: ',value='{1,2,3,4}')

xsstr = ''


for i in xxs:

    if i != '{' and i != '}' and i != '[' and i != ']' and i != '(' and i != ')' and i != ' ':
        xsstr = xsstr + i

fxxs = st.text_input('Ingrese los valores de f(x): ',value='{1,2.14557,3.141592,4}')

x = list(map(float,xsstr.split(',')))
intstrr = ''




for t in fxxs:

    if t != '{' and t != '}' and t != '[' and t != ']' and t != '(' and t != ')' and t != ' ':
        intstrr = intstrr + t

fx = list(map(float,intstrr.split(',')))


#st.write(x)
#st.write(fx)



method = Newton_interpolation(x,fx)
st.write('_El polinomio de Interpolacion está dado por:_')
st.latex(sy.latex(method[0]))

st.pyplot(method[1])
