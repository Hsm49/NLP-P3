# ---------------------------| Bibliotecas |---------------------------
from tkinter import filedialog                                                  # Cuadro de dialogo
from spacy.lang.es.stop_words import STOP_WORDS                                 # Palabras vacias
import spacy                                                                    # Procesamiento de lenguaje natural
import math                                                                     # Matematicas
import os                                                                       # Rutas ingresadas
import re                                                                       # Expresiones regulares
import numpy                                                                    # Operaciones con matrices complejas
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer    # Vectorizador
import pickle                                                                   # Serialización de objetos
from prettytable import PrettyTable                                             # Generar tablas
# ---------------------| Funciones |---------------------
# ----------------------------| Funciones |----------------------------
def cargar_archivo():
    ruta_archivo = filedialog.askopenfilename(title="Abrir archivo", filetypes=(("Archivos de texto", "*.txt"),))
    nombre_archivo = os.path.splitext(os.path.basename(ruta_archivo))[0]
    numero = re.findall(r'\d+', nombre_archivo)
    with open(ruta_archivo, "r", encoding="utf-8") as informacion:
        texto_completo = informacion.read()  # Leer el contenido completo del archivo
    
    if numero == []:
        return texto_completo, "0"
    else:
        return texto_completo, numero[0]
    
def text_normalization(texto):
    # Cargamos el modelo de Spacy
    nlp = spacy.load("es_core_news_sm")

    doc = nlp(texto)
    tokens_normalizados = []
    for token in doc:
        # Filtrar stop words y lematizar
        if token.text.lower() not in STOP_WORDS:
            tokens_normalizados.append(token.lemma_)
    # Unir los tokens normalizados en una sola cadena de texto
    texto_normalizado = ' '.join(tokens_normalizados)
    return texto_normalizado

def vectorize(corpus : list, vectorizer_token_pattern : str, vector_representation_type : int, vector_n_gram_range : tuple):
        x = None
        if(vector_representation_type == 2):

            #Crear vectorizador de tfidf
            tfidf_vectorizer = TfidfVectorizer(ngram_range=vector_n_gram_range,token_pattern=vectorizer_token_pattern)
            x = tfidf_vectorizer.fit_transform(corpus) #Realizar fit y transform al vectorizador de tf-idf
            print("Vectorizador: \n",tfidf_vectorizer,"\n\nVector: \n", x.toarray(),"\n\n")
            return (tfidf_vectorizer,x) #Devolver tupla con el vectorizador tfidf y el vector X

        #Si la representación no es de tipo TF-IDF entonces es de frequencia o binarizado    
        else:
            is_binary = None #Intuir que el Tipo de representación es ninguno
            #Verificar que el tipo de representación sea de tipo 0
            if(vector_representation_type == 0):
                is_binary = False
            #Verificar que el tipo de representación sea de tipo 1    
            elif(vector_representation_type == 1):
                is_binary = True  
            #En caso de que la representación sea una que no haya sido establecida    
            else:
                print("Tipo de representación desconocido")
                return None;    

            #Crear vectorizador de conteo
            count_vectorizer = CountVectorizer(binary=is_binary,ngram_range=vector_n_gram_range,token_pattern=vectorizer_token_pattern)
            x = count_vectorizer.fit_transform(corpus) #Realizar fit y transform al vectorizador de conteo
            print("Vectorizador: \n",count_vectorizer,"\n\nVector: \n", x.toarray(),"\n\n")
            return (count_vectorizer,x) #Devolver tupla con el vectorizador de contador  y el vector X

# Elegir un arreglo de vectores para comparar
def tomar_vector(op, vectors):
    documents = None
    vectorizator = None
    config = None
    if op == ("Título", "Unigramas", "Frecuencia"):
        documents = vectors[0][1]
        vectorizator = vectors[0][0]
        config = ("Título", "Unigramas", "Frecuencia")
    elif op == ("Título", "Unigramas", "Binarizado"):
        documents = vectors[1][1]
        vectorizator = vectors[1][0]
        config = ("Título", "Unigramas", "Binarizado")
    elif op == ("Título", "Unigramas", "TF-IDF"):
        documents = vectors[2][1]
        vectorizator = vectors[2][0]
        config = ("Título", "Unigramas", "TF-IDF")
    elif op == ("Título", "Bigramas", "Frecuencia"):
        documents = vectors[3][1]
        vectorizator = vectors[3][0]
        config = ("Título", "Bigramas", "Frecuencia")
    elif op == ("Título", "Bigramas", "Binarizado"):
        documents = vectors[4][1]
        vectorizator = vectors[4][0]
        config = ("Título", "Bigramas", "Binarizado")
    elif op == ("Título", "Bigramas", "TF-IDF"):
        documents = vectors[5][1]
        vectorizator = vectors[5][0]
        config = ("Título", "Bigramas", "TF-IDF")
    elif op == ("Contenido", "Unigramas", "Frecuencia"):
        documents = vectors[6][1]
        vectorizator = vectors[6][0]
        config = ("Contenido", "Unigramas", "Frecuencia")
    elif op == ("Contenido", "Unigramas", "Binarizado"):
        documents = vectors[7][1]
        vectorizator = vectors[7][0]
        config = ("Contenido", "Unigramas", "Binarizado")
    elif op == ("Contenido", "Unigramas", "TF-IDF"):
        documents = vectors[8][1]
        vectorizator = vectors[8][0]
        config = ("Contenido", "Unigramas", "TF-IDF")
    elif op == ("Contenido", "Bigramas", "Frecuencia"):
        documents = vectors[9][1]
        vectorizator = vectors[9][0]
        config = ("Contenido", "Bigramas", "Frecuencia")
    elif op == ("Contenido", "Bigramas", "Binarizado"):
        documents = vectors[10][1]
        vectorizator = vectors[10][0]
        config = ("Contenido", "Bigramas", "Binarizado")
    elif op == ("Contenido", "Bigramas", "TF-IDF"):
        documents = vectors[11][1]
        vectorizator = vectors[11][0]
        config = ("Contenido", "Bigramas", "TF-IDF")
    elif op == ("Título + Contenido", "Unigramas", "Frecuencia"):
        documents = vectors[12][1]
        vectorizator = vectors[12][0]
        config = ("Título + Contenido", "Unigramas", "Frecuencia")
    elif op == ("Título + Contenido", "Unigramas", "Binarizado"):
        documents = vectors[13][1]
        vectorizator = vectors[13][0]
        config = ("Título + Contenido", "Unigramas", "Binarizado")
    elif op == ("Título + Contenido", "Unigramas", "TF-IDF"):
        documents = vectors[14][1]
        vectorizator = vectors[14][0]
        config = ("Título + Contenido", "Unigramas", "TF-IDF")
    elif op == ("Título + Contenido", "Bigramas", "Frecuencia"):
        documents = vectors[15][1]
        vectorizator = vectors[15][0]
        config = ("Título + Contenido", "Bigramas", "Frecuencia")
    elif op == ("Título + Contenido", "Bigramas", "Binarizado"):
        documents = vectors[16][1]
        vectorizator = vectors[16][0]
        config = ("Título + Contenido", "Bigramas", "Binarizado")
    elif op == ("Título + Contenido", "Bigramas", "TF-IDF"):
        documents = vectors[17][1]
        vectorizator = vectors[17][0]
        config = ("Título + Contenido", "Bigramas", "TF-IDF")
    return documents, vectorizator, config

def generar_diccionario(vectores):
    diccionario = {}
    count = 1
    for vector in vectores:
        diccionario[count] = vector.toarray().tolist()[0]
        count += 1
    return diccionario

def cosine(x, y):
    # Calcular el producto punto entre x y y
    val = sum(x[index] * y[index] for index in range(len(x)))
    # Calcular las normas de x y y
    sr_x = math.sqrt(sum(x_val**2 for x_val in x))
    sr_y = math.sqrt(sum(y_val**2 for y_val in y))
    # Evitar división por cero si alguna de las normas es cero
    if sr_x == 0 or sr_y == 0:
        return 0  # O manejar de otra manera si es necesario
    # Calcular la similitud de coseno
    res = val / (sr_x * sr_y)
    return res

def calcular_similitud(myDoc, allDoc):
    resultados = {}
    count = 1
    for doc, vector in allDoc.items():
        resultados[count] = cosine(myDoc['unique'], vector)
        count += 1
    return resultados

def ordenar_similitud(similitud):
    # Ordenar el diccionario de similitud en función de los valores (similitudes) en orden descendente
    similitud_ordenada = sorted(similitud.items(), key=lambda x: x[1], reverse=True)
    return similitud_ordenada

def agregar_filas(table, results_keys, results_values, configurations):
    for _ in range(10):
        table.add_row([f"{results_keys[_]}", f"{configurations[2]}", f"{configurations[1]}", f"{configurations[0]}", f"{(results_values[_] * 100):.2f}%"])
    return table
