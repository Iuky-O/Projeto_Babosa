from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Carregar o modelo pré-treinado
model = load_model('model.h5')

class_names = ['healthy_leaf', 'rot', 'rust'] 
class_names_pt = ['folha_saudável', 'podre', 'ferrugem']  

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/uploads/<filename>")
def upload_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Receber a imagem enviada
        file = request.files["file"]
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(img_path)

        # Carregar e processar a imagem
        img = image.load_img(img_path, target_size=(244, 244))  # Ajuste para o tamanho de entrada do modelo
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  
        img_array /= 255.0  # Normaliza a imagem

        # Fazer a previsão
        preds = model.predict(img_array)
        print(preds)  # Imprime as probabilidades de cada classe

        # Obter a classe com maior probabilidade
        predicted_class = np.argmax(preds, axis=1)
        print(f'Classe prevista: {predicted_class}')  # Imprime a classe com maior probabilidade

        # Mapear o índice da classe para o nome da classe
        predicted_label = class_names[predicted_class[0]]
        predicted_label_pt = class_names_pt[predicted_class[0]]

        return render_template("index.html", 
                               prediction_en=predicted_label, 
                               prediction_pt=predicted_label_pt, 
                               image_path=img_path)
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
