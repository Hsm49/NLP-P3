import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from funciones import *

# ----------------------| Precargar información |----------------------
DUMP_FILENAME = "P3_2/routes/Dumps/vectors_dump.pkl"
# Abrimos el archivo de Dumpeo
with open(DUMP_FILENAME,"rb") as dump_file:
    vectores = tuple(pickle.load(dump_file)) #Cargar lista de vectores

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

def ejecutar_proceso():
    texto_usuario, _ = cargar_archivo()
    texto_norm = text_normalization(texto_usuario)
    options = (cont_var.get(), extr_var.get(), vec_var.get())
    documents, vectorizador, configuracion = tomar_vector(options, vectores)
    document = vectorizador.transform([texto_norm])
    data_document = {}
    data_document['unique'] = document[0].toarray().tolist()[0]
    data_documents = generar_diccionario(documents)
    similitud_coseno = calcular_similitud(data_document, data_documents)
    similitud_ordenada = ordenar_similitud(similitud_coseno)
    claves = []
    valores = []
    limite = 1
    for clave, valor in similitud_ordenada:
        claves.append(clave)
        valores.append(valor)
        if limite >= 10:
            break
        limite += 1

    tabla = PrettyTable()
    tabla.field_names = ["Corpus Document", "Vector Representation", "Extracted Features", "Comparison Element", "Similarity Value"]
    tabla = agregar_filas(tabla, claves, valores, configuracion)
    tabla_text.config(state=tk.NORMAL)
    tabla_text.delete('1.0', tk.END)
    tabla_text.insert(tk.END, str(tabla))
    tabla_text.config(state=tk.DISABLED)

# Crear la ventana principal
root = tk.Tk()
root.title("Práctica 3 NLP")
root.geometry("1280x720")  # Tamaño inicial de la ventana
root.configure(bg="#E20071")

# Cambiar el color de fondo de la tabla a negro
style = ttk.Style()
style.configure("Tabla.TFrame", background="black")

# Crear y posicionar los widgets
cont_label = ttk.Label(root, text="Contenido:", font=("Times New Roman", 14, "bold"))
cont_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
cont_var = tk.StringVar(value="Título")
cont_combobox = ttk.Combobox(root, textvariable=cont_var, values=["Título", "Contenido", "Título + Contenido"], state="readonly", font=("Times New Roman", 14, "bold"))
cont_combobox.grid(row=0, column=1, padx=5, pady=5)

extr_label = ttk.Label(root, text="Características Extraídas:", font=("Times New Roman", 14, "bold"))
extr_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
extr_var = tk.StringVar(value="Unigramas")
extr_combobox = ttk.Combobox(root, textvariable=extr_var, values=["Unigramas", "Bigramas"], state="readonly", font=("Times New Roman", 14, "bold"))
extr_combobox.grid(row=1, column=1, padx=5, pady=5)

vec_label = ttk.Label(root, text="Tipo de Vector:", font=("Times New Roman", 14, "bold"))
vec_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
vec_var = tk.StringVar(value="Frecuencia")
vec_combobox = ttk.Combobox(root, textvariable=vec_var, values=["Frecuencia", "Binarizado", "TF-IDF"], state="readonly", font=("Times New Roman", 14, "bold"))
vec_combobox.grid(row=2, column=1, padx=5, pady=5)

# Establecer el estilo del botón
style.configure("Boton.TButton", font=("Times New Roman", 14, "bold"))
ejecutar_button = ttk.Button(root, text="Seleccionar texto", command=ejecutar_proceso, style="Boton.TButton")
ejecutar_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

tabla_text = tk.Text(root, bg="black", fg="white", font=("Consolas", 16))  # Cambiar color de fondo y texto
tabla_text.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

# Establecer el color de fondo de la tabla
tabla_frame = ttk.Frame(tabla_text, style="Tabla.TFrame")
tabla_frame.grid(row=0, column=0, sticky="nsew")

# Establecer el color de fondo de la tabla
tabla_text.window_create("end", window=tabla_frame)

# Configurar el tamaño de las filas y columnas para que se expandan
root.grid_rowconfigure(4, weight=1)
root.grid_columnconfigure(0, weight=1)

# Ejecutar el bucle de eventos de la ventana
root.mainloop()
