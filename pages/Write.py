import streamlit as st
import pandas as pd
import numpy as np

"""
Podemos asignar un header especifico para nuestra pagina a traves de st.header.
El header(encabezado) de la pagina puede tener colores, emojis y estilos de letras
para hecer uso de ellos necesitamos introducir una secuencia especifica de caracteres
los cuales determinan el color, el estilo de la letra o un emoji especifico.

"""
st.header('Un encabezado con  _italica_ :blue[color] y emojis :sunglasses:')

"""
st.write nos permite imprimir cualquier cosa en la pagina
de igual forma  puede tener colores, emojis y estilos de letras
"""
#ejemplo 1
st.write('_Hello_  :blue[World] :sunglasses:')

#ejemplo 2
st.write(1234)

"""
Usando un dataframe de pandas podemos de igual forma imprimirlo usando
st.write, en este caso se imprimira los datos como una tabla

Nota:
Un dataframe de pandas es estructura de datos de dos dimensiones (rectangulares)
que pueden contener datos de diferentes tipos.
Para crear un data frame necesitamos un conjunto de datos de dos dimensiones (Matriz)
y una etiqueta para cada columna (string) de la matriz, para ello podemos crear el data frame
usando un diccionario con los nombres de las columnas como llaves y la respectiva columna asociada a la
llave como un array unidimensional (ejemplo #3) o de igual forma podemos pasar la matriz y con el parametro columns
asignar la etiqueta para cada columna  de la matriz automaticamente (ejemplo #4)
"""
#ejemplo 3
d1 = pd.DataFrame({
    'Primer columna' : [1,2,3],
    'Segunda columna' : [4,4,4]

     }
)


st.write(d1)
#ejemplo 4
d2 = pd.DataFrame(np.random.randn(200,3),columns=['Primer columna', 'Segunda columna', 'Tercer columna'])


st.write(d2)


"""
Analogamente podemos desplegar textos escritos en latex para ello hacemos uso de st.latex pasando la formula escrita
en latex.
"""

st.latex(r'''
...     a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
...     \sum_{k=0}^{n-1} ar^k =
...     a \left(\frac{1-r^{n}}{1-r}\right)
...     ''')


"""
De igual forma podemos desplegar texto en Markdown  con st.markdown.
"""

st.markdown(":green[$sqrt{x^2+y^2}=1$] is a Pythagorean identity. :pencil:")

"""
Finalmente podemos desplegar codigo de cualquier lenguage a traves de st.code.
Pasando el codigo como un string y definiendo el lenguaje utilizado con el parametro language.
A continuacion presentamos el codigo usado para crear esta pagina.
"""

codee = '''
import streamlit as st
import pandas as pd
import numpy as np

"""
Podemos asignar un header especifico para nuestra pagina a traves de st.header.
El header(encabezado) de la pagina puede tener colores, emojis y estilos de letras
para hecer uso de ellos necesitamos introducir una secuencia especifica de caracteres
los cuales determinan el color, el estilo de la letra o un emoji especifico.

"""
st.header('Un encabezado con  _italica_ :blue[color] y emojis :sunglasses:')

"""
st.write nos permite imprimir cualquier cosa en la pagina
de igual forma  puede tener colores, emojis y estilos de letras
"""
#ejemplo 1
st.write('_Hello_  :blue[World] :sunglasses:')

#ejemplo 2
st.write(1234)

"""
Usando un dataframe de pandas podemos de igual forma imprimirlo usando
st.write, en este caso se imprimira los datos como una tabla

Nota:
Un dataframe de pandas es estructura de datos de dos dimensiones (rectangulares)
que pueden contener datos de diferentes tipos.
Para crear un data frame necesitamos un conjunto de datos de dos dimensiones (Matriz)
y una etiqueta para cada columna (string) de la matriz, para ello podemos crear el data frame
usando un diccionario con los nombres de las columnas como llaves y la respectiva columna asociada a la
llave como un array unidimensional (ejemplo #3) o de igual forma podemos pasar la matriz y con el parametro columns
asignar la etiqueta para cada columna  de la matriz automaticamente (ejemplo #4)
"""
#ejemplo 3
d1 = pd.DataFrame({
    'Primer columna' : [1,2,3],
    'Segunda columna' : [4,4,4]

     }
)


st.write(d1)
#ejemplo 4
d2 = pd.DataFrame(np.random.randn(200,3),columns=['Primer columna', 'Segunda columna', 'Tercer columna'])


st.write(d2)


"""
Analogamente podemos desplegar textos escritos en latex para ello hacemos uso de st.latex pasando la formula escrita
en latex.
"""

st.latex(r' ''
...     a + ar + a r^2 + a r^3 + \\cdots + a r^{n-1} =
...     \\sum_{k=0}^{n-1} ar^k =
...     a \\left(\frac{1-r^{n}}{1-r}\right)
...     ' '')


"""
De igual forma podemos desplegar texto en Markdown  con st.markdown.
"""

st.markdown(":green[$sqrt{x^2+y^2}=1$] is a Pythagorean identity. :pencil:")

"""
Finalmente podemos desplegar codigo de cualquier lenguage a traves de st.code.
Pasando el codigo como un string y definiendo el lenguaje utilizado con el parametro language.
A continuacion presentamos el codigo usado para crear esta pagina.
"""

codee = ' ' '

' ' '

st.code(codee,language='python')


'''

st.code(codee,language='python')
