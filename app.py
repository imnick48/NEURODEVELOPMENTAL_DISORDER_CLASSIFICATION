from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers
from tensorflow.keras.utils import custom_object_scope
import numpy as np
import os
from werkzeug.utils import secure_filename
import tensorflow as tf

app = Flask(__name__)

# ✅ Upload settings
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Custom Layer for Vision Transformer
class ClassTokenLayer(layers.Layer):
    def __init__(self, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim
        self.class_token = tf.Variable(tf.zeros([1, 1, projection_dim]))

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        class_tokens = tf.broadcast_to(self.class_token, [batch_size, 1, self.projection_dim])
        return layers.Concatenate(axis=1)([class_tokens, inputs])

    def get_config(self):
        config = super().get_config()
        config.update({"projection_dim": self.projection_dim})
        return config

# ✅ Classification mapping (model path + class labels)
CLASSIFICATION_MODELS = {
    "dysgraphia_vs_normal": {
        "path": "app/model/DysgraphiaVsNormal/vision_transformer.keras",
        "classes": ["Dysgraphia", "Normal"]
    },
    "low_vs_potential": {
        "path": "app/model/LowVsPotential/vision_transformer.keras",
        "classes": ["LowPotential", "Potential"]
    },
    "potential_vs_normal": {
        "path": "app/model/NormalvsHighPotential/vision_transformer_with_data_aug.keras",
        "classes": ["Normal", "Potential"]
    },
    "normal_vs_low": {
        "path": "app/model/NormalvsLowPotential/vt_normalvslow.keras",
        "classes": ["LowPotential", "Normal"]
    }
}

# ✅ Cache for loaded models to avoid reloading
loaded_models = {}

# ✅ Allowed file check
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ✅ Load model with caching
def get_model(classification_key):
    if classification_key in loaded_models:
        return loaded_models[classification_key]

    model_path = CLASSIFICATION_MODELS[classification_key]["path"]
    with custom_object_scope({'ClassTokenLayer': ClassTokenLayer}):
        model = load_model(model_path, compile=False)
    loaded_models[classification_key] = model
    return model

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        classification = request.form.get('classification')
        if not classification or classification not in CLASSIFICATION_MODELS:
            return render_template('index.html', error_message='Please select a valid classification type.')

        # ✅ Get model and class labels
        model_info = CLASSIFICATION_MODELS[classification]
        classes = model_info["classes"]

        # ✅ Load model (cached)
        try:
            model = get_model(classification)
            print(f"✅ Using model for {classification}")
        except Exception as e:
            return render_template('error.html', message=f"Error loading model: {e}")

        # ✅ Check for image upload
        if 'image' not in request.files:
            return render_template('index.html', error_message='No file selected.')

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error_message='No selected file.')

        if file and allowed_file(file.filename):
            try:
                # ✅ Save image
                filename = secure_filename(file.filename)
                full_save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(full_save_path)

                relative_static_path_url = os.path.join('uploads', filename).replace('\\', '/')

                # ✅ Preprocess image (256x256)
                img = load_img(full_save_path, target_size=(256, 256))
                img_array = img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # ✅ Predict
                preds = model.predict(img_array)
                predicted_class_index = np.argmax(preds)
                prediction_label = classes[predicted_class_index]
                confidence = round(float(np.max(preds)) * 100, 2)

                return render_template(
                    'result.html',
                    prediction=prediction_label,
                    confidence=confidence,
                    image_path=relative_static_path_url,
                    selected_classification=classification
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
    app.run(debug=True)
