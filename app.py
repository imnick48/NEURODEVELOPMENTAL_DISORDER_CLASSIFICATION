from flask import Flask, render_template, request
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
#3_layer_cnn_with_data_aug
# Path to the .keras model
MODEL_PATH = 'app/model/lenet_with_data_aug.keras'
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = None
try:
    model = load_model(MODEL_PATH, compile=False)
    print(f"Model loaded successfully from {MODEL_PATH}.")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    print("Ensure the model file exists at the correct path.")

# Fixed class order
classes = ['Low Potential Dysgraphia', 'Potential Dysgraphia']
print(f"Model expects classes in this order: {classes}")

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if model is None:
        return render_template('error.html', message="Model could not be loaded. Please try again later.")

    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error_message='No file selected.')

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error_message='No selected file.')

        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                full_save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(full_save_path)
                print(f"File saved successfully to {full_save_path}")

                relative_static_path_url = os.path.join('uploads', filename).replace('\\', '/')

                img = load_img(full_save_path, target_size=(224, 224))
                img_array = img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                preds = model.predict(img_array)
                predicted_class_index = np.argmax(preds)
                prediction_label = classes[predicted_class_index]
                print(classes[predicted_class_index])

                return render_template(
                    'result.html',
                    prediction=prediction_label,
                    image_path=relative_static_path_url
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                return render_template('error.html', message=f"An error occurred during image processing: {e}")

        else:
            return render_template('index.html', error_message='Invalid file type. Please upload PNG, JPG, JPEG, or GIF.')

    return render_template('index.html')

@app.route('/error')
def error_page():
    message = request.args.get('message', 'An unknown error occurred.')
    return render_template('error.html', message=message)

if __name__ == '__main__':
    if model is None:
        print("\nFlask app will not start because the model could not be loaded.")
    else:
        app.run(debug=True)
