from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import layers
from tensorflow.keras.utils import custom_object_scope
import numpy as np
import os
import io
import base64
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Custom Layer for Vision Transformer
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

# Available model architectures
MODEL_ARCHITECTURES = {
    "vision_transformer": "Vision Transformer",
    "lenet5": "LeNet-5",
    "cnn": "CNN (3-Layer)",
}

# Classification mapping (class labels + model paths per architecture)
# lenet5 and cnn use .tflite, vision_transformer stays as .keras
CLASSIFICATION_MODELS = {
    "dysgraphia_vs_normal": {
        "classes": ["Dysgraphia", "Normal"],
        "models": {
            "vision_transformer": "app/model/DysgraphiavsNormal/normal_vs_dysgraphia_vision_transformer.keras",
            "lenet5": "app/model/DysgraphiavsNormal/lenet_with_data_aug(Normal Vs dys).tflite",
            "cnn":    "app/model/DysgraphiavsNormal/3_layer_cnn_with_data_aug(Normal Vs Dys).tflite",
        }
    },
    "potential_vs_normal": {
        "classes": ["Normal", "Potential"],
        "models": {
            "vision_transformer": "app/model/NormalvsHighPotential/vision_transformer_with_data_aug.tflite",
            "lenet5": "app/model/NormalvsHighPotential/lenet_with_data_aug (Normal VsHigh).tflite",
            "cnn":    "app/model/NormalvsHighPotential/3_layer_cnn_with_data_aug(Normal vs HighPotential).tflite",
        }
    },
    "normal_vs_low": {
        "classes": ["LowPotential", "Normal"],
        "models": {
            "vision_transformer": "app/model/NormalvsLowPotential/vt_normalvslow.tflite",
            "lenet5": "app/model/NormalvsLowPotential/lenet_with_data_aug(Normal vs Low).tflite",
            "cnn":    "app/model/NormalvsLowPotential/3_layer_cnn_with_data_aug(Normal vs Low).tflite",
        }
    },
    "3Class": {
        "classes": ["Potential", "LowPotential", "Normal"],
        "models": {
            "vision_transformer": "app/model/3Class/Vision_Transformer_3class.tflite",
            "lenet5": "app/model/3Class/lenet5(3 class).tflite",
            "cnn":    "app/model/3Class/cnn(3 class).tflite",
        }
    }
}

# Cache for loaded models — stores ('tflite', interpreter) or ('keras', model)
loaded_models = {}

# Allowed file check
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load model with caching — handles both .tflite and .keras
def get_model(classification_key, model_key):
    cache_key = (classification_key, model_key)
    if cache_key in loaded_models:
        return loaded_models[cache_key]

    model_path = CLASSIFICATION_MODELS[classification_key]["models"][model_key]

    if model_path.endswith('.tflite'):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        loaded_models[cache_key] = ('tflite', interpreter)
    else:
        with custom_object_scope({'ClassTokenLayer': ClassTokenLayer}):
            model = load_model(model_path, compile=False)
        loaded_models[cache_key] = ('keras', model)

    return loaded_models[cache_key]

# Run inference — works for both tflite and keras models
def run_prediction(model_tuple, img_array):
    model_type, model_obj = model_tuple

    if model_type == 'tflite':
        input_details = model_obj.get_input_details()
        output_details = model_obj.get_output_details()
        model_obj.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
        model_obj.invoke()
        preds = model_obj.get_tensor(output_details[0]['index'])
    else:
        preds = model_obj.predict(img_array)

    return preds

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        classification = request.form.get('classification')
        model_architecture = request.form.get('model_architecture')

        if not classification or classification not in CLASSIFICATION_MODELS:
            return render_template('index.html', error_message='Please select a valid classification type.')

        if not model_architecture or model_architecture not in MODEL_ARCHITECTURES:
            return render_template('index.html', error_message='Please select a valid model architecture.')

        # Get class labels
        model_info = CLASSIFICATION_MODELS[classification]
        classes = model_info["classes"]

        # Load model (cached by classification + architecture)
        try:
            model_tuple = get_model(classification, model_architecture)
            print(f"Using {MODEL_ARCHITECTURES[model_architecture]} model for {classification}")
        except Exception as e:
            return render_template('error.html', message=f"Error loading model: {e}")

        # Check for image upload
        if 'image' not in request.files:
            return render_template('index.html', error_message='No file selected.')

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error_message='No selected file.')

        if file and allowed_file(file.filename):
            try:
                # Read image directly from stream (no disk save)
                img = Image.open(file.stream).convert('RGB')

                # Encode original image as base64 for display
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                buf.seek(0)
                image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

                # Preprocess image (256x256)
                img_resized = img.resize((256, 256))
                img_array = img_to_array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Predict
                preds = run_prediction(model_tuple, img_array)
                predicted_class_index = np.argmax(preds)
                prediction_label = classes[predicted_class_index]
                confidence = round(float(np.max(preds)) * 100, 2)

                return render_template(
                    'result.html',
                    prediction=prediction_label,
                    confidence=confidence,
                    image_base64=image_base64,
                    selected_classification=classification,
                    selected_model=MODEL_ARCHITECTURES[model_architecture]
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