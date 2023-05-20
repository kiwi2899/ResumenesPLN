import os
from flask import Flask, render_template, request, send_file, flash, make_response
import nltk
import pandas as pd
import numpy as np
from sentence_splitter import SentenceSplitter
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
app.secret_key = 'secret_key'

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/resumen', methods=['POST'])
def resumen():
    archivo = request.files['archivo']
    porcentaje = int(request.form['porcentaje'])

    # Guardar el archivo en una ruta temporal
    archivo_path = 'temp.txt'
    archivo.save(archivo_path)

    splitter = SentenceSplitter(language='es')

    # Leer el contenido del archivo
    with open(archivo_path, 'r', encoding='utf-8') as file:
        contenido = file.read()

    # Obtener las oraciones
    oraciones = splitter.split(contenido)

    nltk.download("punkt")
    nltk.download("stopwords")
    
    palabras_funcionales = nltk.corpus.stopwords.words("spanish")
    print(palabras_funcionales)

    allTokens = []
    for oracion in oraciones:
        # Separar oraciones en segmentos más cortos usando puntos, comas y otros caracteres de puntuación
        segmentos = nltk.sent_tokenize(oracion, "spanish")
        allTokens.extend(segmentos)

    vectorizer = TfidfVectorizer(stop_words=palabras_funcionales)
    X = vectorizer.fit_transform(allTokens)  # todos los tokens de todas las oraciones
    vocabulario = vectorizer.get_feature_names_out()
    tabla_frecuencias = pd.DataFrame(X.toarray(), index=allTokens, columns=vocabulario)
    print(tabla_frecuencias)
    promedio = np.mean(tabla_frecuencias, axis=1)
    print(promedio)
    mas_importantes = promedio.sort_values(ascending=False)
    print(mas_importantes)

    cantidad_oraciones = round(len(mas_importantes) * porcentaje / 100)
    print(cantidad_oraciones)

    if cantidad_oraciones < 1:
        flash('No hay suficientes oraciones para particionar según el porcentaje seleccionado.')
        return render_template('index.html')

    resumen = ' '.join(mas_importantes.head(cantidad_oraciones).index)
    
    # Separar las oraciones en el resultado final con saltos de línea
    resumen = resumen.replace('. ', '.\n')

    return render_template('resumen.html', resumen=resumen)

if __name__ == '__main__':
    app.run()


