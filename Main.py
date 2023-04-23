import streamlit as st
import pandas as pd
import numpy as np
import sympy as sy
from matplotlib import pyplot as plt
from sympy.plotting.plot import MatplotlibBackend, Plot
from sympy.plotting import plot3d,plot3d_parametric_line
import plotly as ply


def parse_inputsys(inp):
    eq = []
    sfix = ''

    for i in inp:
        if i != '{' and i != '}' and i != '[' and i != ']' and i != '(' and i != ')' and i != ' ':
            sfix = sfix + i

    for t in sfix.split(','):
        eq.append(sy.parse_expr(t,transformations='all'))

    return eq

def get_maxsymbs(sys):
    maxs = 0
    s = []
    for i in sys:
        if max(maxs,len(i.free_symbols)) != maxs:
            s = list(i.free_symbols)

    return s

def jacobian(ff,symb):
    """
    It takes a vector of functions and a vector of symbols and returns the Jacobian matrix of the functions with respect to
    the symbols
    :param ff: the function
    :param symb: the symbols that are used in the function
    :return: A matrix of the partial derivatives of the function with respect to the variables.
    """
    m = []

    for i in range(0,len(ff)):
        aux  = []
        for j in range(0,len(symb)):
            aux.append(sy.diff(ff[i],symb[j]))
        m.append(aux)

    return np.array(m)

def hessian(ff,symb):
    """
    It takes a vector of functions and a vector of symbols and returns the Hessian matrix of the functions with respect to
    the symbols
    :param ff: a list of functions of the form f(x,y,z)
    :param symb: the symbols that are used in the function
    :return: A matrix of the second derivatives of the function.
    """

    m = []

    for i in range(0,len(ff)):
        aux  = []
        for j in range(0,len(symb)):
            aux.append(sy.diff(ff[i],symb[j],2))
        m.append(aux)
    return np.array(m)

def eval_matrix(matrix , v,symb):
    """
    It takes a matrix, a list of symbols and a list of values, and returns the matrix with the symbols substituted by the
    values

    :param matrix: the matrix of the system of equations
    :param v: the vector of values for the variables
    :param symb: the symbols that will be used in the matrix
    :return: the matrix with the values of the variables substituted by the values of the vector v.
    """
    e = 0
    mm = []
    for i in range(0,len(matrix)):
        aux = []
        ev = []
        for k in range(0,len(symb)):
            ev.append((symb[k],v[k]))
        for j in range(len(matrix[i])):
            aux.append(matrix[i][j].subs(ev).evalf())
        mm.append(aux)
    return np.array(mm)

def evalVector(ff, x0,symb):
    """
    > Given a list of symbolic expressions, a list of values for the symbols, and a list of the symbols, evaluate the
    symbolic expressions at the given values

    :param ff: the vector of functions
    :param x0: initial guess
    :param symb: the symbols that are used in the symbolic expression
    :return: the value of the function at the point x0.
    """
    v = []
    for i in range(0,len(ff)):
        ev = []

        for k in range(0,len(x0)):
            ev.append((symb[k],x0[k]))

        v.append(ff[i].subs(ev).evalf())
    return np.array(v)

def NewtonMethod( ff, x0,symb ):
    """
    The function takes in a vector of functions, a vector of initial guesses, and a vector of symbols. It then calculates
    the Jacobian matrix, the Jacobian matrix evaluated at the initial guess, the inverse of the Jacobian matrix evaluated at
    the initial guess, the vector of functions evaluated at the initial guess, and then the Newton step.

    The function returns the Newton step.

    :param ff: the function we want to find the root of
    :param x0: initial guess
    :param symb: the symbols used in the function
    :return: The return value is the x_np1 value.
    """
    j = jacobian(ff,symb)
    #print("Jacobian Matrix")
    #pprint(Matrix(j))
    jev = sy.Matrix( eval_matrix(j,x0,symb))
    #print("J(",x0,")")
    #pprint(jev)

    jinv = jev.inv()
    #print("F(",x0,")")
    ffev = sy.Matrix(evalVector(np.transpose(ff),x0,symb))
    #print("J^-1(",x0,")*","F(",x0,")")
    mm = sy.Matrix(jinv)*ffev
    #pprint(mm)
    x_np1 = sy.Matrix(np.transpose(np.array(x0)))
    #pprint(x_np1-mm)
    return list(x_np1-mm)

def norm_inf(x_0,x_1):
    """
    > The function `norm_inf` takes two vectors `x_0` and `x_1` and returns the maximum absolute difference between the two
    vectors

    :param x_0: the initial guess
    :param x_1: the vector of the current iteration
    :return: The maximum difference between the two vectors.
    """
    a = [abs(x_1[i]-x_0[i]) for i in range(len(x_0))]
    return max(a)

def newton_method(ff,x_0,symbs,error,maxiter):
    """
    Given a function (x,y)$, a starting point $, and a list of symbols,
    the function will return the next point $ in the Newton's method sequence

    :param ff: the function to be minimized
    :param x_0: initial guess
    :param symbs: the symbols that we're using in the function
    :return: the final value of x_0, the list of x values, and the list of y values.
    """
    #pprint(Matrix(x_0))
    xs = []
    ys = []
    xns = [x_0]
    erros = []
    iterr = 0
    while True and iterr < maxiter:

        x_1 = NewtonMethod(ff,x_0,symbs)
        #print(x_1)
        ninf = norm_inf(x_0,x_1)
        erros.append(ninf)
        #print(ninf)

        x_0 = list(x_1)
        xns.append(tuple(x_0))
        xs.append(x_0[0])
        ys.append(x_0[1])
        if ninf < error:
            #print("Iteraciones: ",iterr)
            break
        iterr = iterr+1

    #print(x_0)
    return xns,xs,ys,erros


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


st.title(':blue[Metodo de Newton]')

st.subheader(':blue[Descripción del método]')

st.subheader(':blue[Ejemplo]')








st.subheader('Método')
sys = st.text_input('Ingrese el sistema de ecuaciones ',value=r'{x^2+3*y*x+2,y**3*x**2-2*y**3-5}')
try:
    system = parse_inputsys(sys)
except :
    st.error('Error al introducir el sistema de ecuaciones', icon="🚨")

st.write('_El sistema de ecuaciones es:_')
psys = r'''
    \begin{cases}

'''
for i in system:
    psys = psys + sy.latex(i)
    psys = psys + r' = 0\\'

psys = psys + r'\end{cases}'

st.latex(psys)

try:
    st.write('_Grafica del sistema como función implicita_ : ')
    x,y = sy.symbols('x,y')
    p1 = sy.plot_implicit(system[0],(x,-10,10),(y,-10,10),show=False,line_color='red')
    for i in range(1,len(system)):
        p1.append(sy.plot_implicit(system[i],(x,-10,10),(y,-10,10),show=False,line_color='blue')[0])
    t2dplot =  get_sympy_subplots(p1)
    st.pyplot(t2dplot)

    st.write('_Grafica 3D del sistema_')

    p13d = plot3d(system[0],(x,-10,10),(y,-10,10),show=False,surface_color='r')
    for i in range(1,len(system)):
        p13d.append(plot3d(system[i],(x,-10,10),(y,-10,10),show=False,surface_color='g')[0])

    t3dplot = get_sympy_subplots(p13d)
    st.pyplot(t3dplot)

except Exception as excep:
    st.error(excep)
    st.error('Error al graficar', icon="🚨")


initaprx = st.text_input('Ingrese una aproximacion inicial: ',value=[-1,1])

intaprox = []
intstr = ''




for i in initaprx:

    if i != '{' and i != '}' and i != '[' and i != ']' and i != '(' and i != ')' and i != ' ':
        intstr = intstr + i

try:
    st.write('La aproximacion inicial es: ')
    intaprox = list(map(int, intstr.split(',')))
    st.latex(sy.latex(sy.Matrix(list(intaprox))))
except:
    st.error('Error al introducir la aproximación inicial', icon="🚨")


err = st.text_input('Ingrese el error de tolerancia: ',value='0.00001')
try:
    st.write('El error de tolerancia es:', float(err))
except:
    st.error('Error al introducir el error de tolerancia', icon="🚨")


maxiter = st.slider('Maximo de Iteraciones',10,1000,10)


st.write('_Matrix Jacobiana_:')
symbs = (get_maxsymbs(system))


st.latex(sy.latex(sy.Matrix(jacobian(system,symbs))))

method = newton_method(system,list(intaprox),symbs,float(err),maxiter)

tabb = []

for i in range(0,len(method[1])):
    aux = list(method[0][i])
    aux.append(method[3][i])
    tabb.append(aux)

cols = list(map(str, list(symbs)))
cols.append('Error')
#st.write(tabb)
table = pd.DataFrame(tabb,columns=cols)

st.write(table)
try:
    pf =get_sympy_subplots(p1)
    pf.plot(method[1],method[2],'o')
    st.pyplot(pf)
    auxpd = plot3d(system[0],(x,-1+method[1][-1],method[1][-1]+1),(y,-1+method[2][-1],method[2][-1]+1),show=False)
    auxpd.append(plot3d(system[1],(x,-1+method[1][-1],method[1][-1]+1),(y,-1+method[2][-1],method[2][-1]+1),show=False)[0])
    pda = get_sympy_subplots(auxpd)
    zs = []
    for i in range(0,len(method[1])):
        zs.append(system[0].subs({x : method[1][i],y: method[2][i]}).evalf())
    pda.plot(method[1],method[2],zs,'o',markersize=100,color='red')
    st.pyplot(pda)
except Exception as e:
    st.error(str(e))

indexplot = st.slider('Iteracion',0,len(method[1])-1)


fx = sy.lambdify(list(symbs),system[0])

evalfx = [fx(i,i) for i in np.linspace(0,10,100)]
#st.write(evalfx)

#plyyy = ply.graph_objs.Surface(x=np.linspace(0,10,100),y=np.linspace(0,10,100),z=np.array(evalfx))

fig = ply.graph_objs.Figure(data=table)
st.plotly_chart(fig)



