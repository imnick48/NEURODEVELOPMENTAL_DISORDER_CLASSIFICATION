from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import layers
from tensorflow.keras.utils import custom_object_scope
import numpy as np
from PIL import Image
import tensorflow as tf
import threading
import os
import gc
import io
import base64


app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


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


MODEL_ARCHITECTURES = {
    "vision_transformer": "Vision Transformer",
    "lenet5": "LeNet-5",
    "cnn": "CNN (3-Layer)",
}

CLASSIFICATION_MODELS = {
    "low_vs_potential": {
        "classes": ["Low Potential", "Potential"],
        "models": {
            "vision_transformer": "app/model/LowPotentialvsHighPotential/vision_transformer_with_data_aug(lvh).tflite",
            "lenet5": "app/model/LowPotentialvsHighPotential/lenet_with_data_aug(lvh).tflite",
            "cnn":    "app/model/LowPotentialvsHighPotential/3_layer_cnn_with_data_aug(lvh).tflite",
        }
    },
    "dysgraphia_vs_normal": {
        "classes": ["Dysgraphia", "Normal"],
        "models": {
            "vision_transformer": "app/model/DysgraphiavsNormal/vision_transformer_with_data_aug(nvd).keras",
            "lenet5": "app/model/DysgraphiavsNormal/lenet_with_data_aug(nvd).tflite",
            "cnn":    "app/model/DysgraphiavsNormal/3_layer_cnn_with_data_aug(nvd).tflite",
        }
    },
    "potential_vs_normal": {
        "classes": ["Potential", "Normal"],
        "models": {
            "vision_transformer": "app/model/NormalvsHighPotential/vision_transformer_with_data_aug(nvh).keras",
            "lenet5": "app/model/NormalvsHighPotential/lenet_with_data_aug(nvh).tflite",
            "cnn":    "app/model/NormalvsHighPotential/3_layer_cnn_with_data_aug(nvh).tflite",
        }
    },
    "normal_vs_low": {
        "classes": ["LowPotential", "Normal"],
        "models": {
            "vision_transformer": "app/model/NormalvsLowPotential/vision_transformer_with_data_aug(nvl).tflite",
            "lenet5": "app/model/NormalvsLowPotential/lenet_with_data_aug(nvl).tflite",
            "cnn":    "app/model/NormalvsLowPotential/3_layer_cnn_with_data_aug(nvl).tflite",
        }
    },
    "3Class": {
        "classes": ["Potential", "LowPotential", "Normal"],
        "models": {
            "vision_transformer": "app/model/3Class/vision_transformer_with_data_aug(3Class).tflite",
            "lenet5": "app/model/3Class/lenet_with_data_aug(3Class).tflite",
            "cnn":    "app/model/3Class/3_layer_cnn_with_data_aug(3Class).tflite",
        }
    }
}
POOL_SIZE = 3

_model_semaphores: dict[tuple, threading.Semaphore] = {}
_semaphore_lock = threading.Lock()


def _get_semaphore(cache_key: tuple) -> threading.Semaphore:
    if cache_key not in _model_semaphores:
        with _semaphore_lock:
            if cache_key not in _model_semaphores:
                _model_semaphores[cache_key] = threading.Semaphore(POOL_SIZE)
    return _model_semaphores[cache_key]


class _InterpreterContext:
    """Loads a fresh interpreter (TFLite) or Keras model, yields it, then destroys it completely."""
    def __init__(self, model_path: str, semaphore: threading.Semaphore):
        self._model_path = model_path
        self._semaphore  = semaphore
        self._interp     = None
        self._keras_model = None
        self._is_keras   = model_path.endswith('.keras')

    def __enter__(self):
        self._semaphore.acquire()
        if self._is_keras:
            self._keras_model = load_model(
                self._model_path,
                custom_objects={"ClassTokenLayer": ClassTokenLayer}
            )
            return self
        else:
            self._interp = tf.lite.Interpreter(model_path=self._model_path)
            self._interp.allocate_tensors()
            return self._interp

    def predict(self, img_array: np.ndarray) -> np.ndarray:
        """Only used for .keras models (call via context object, not interpreter)."""
        return self._keras_model.predict(img_array, verbose=0)

    def __exit__(self, *_):
        try:
            if self._is_keras:
                del self._keras_model
                self._keras_model = None
            else:
                del self._interp
                self._interp = None
            gc.collect()
            tf.keras.backend.clear_session()
        finally:
            self._semaphore.release()
            print("[mem] Model released and memory cleared.")


def get_interpreter(classification_key: str, model_key: str) -> "_InterpreterContext":
    cache_key  = (classification_key, model_key)
    model_path = CLASSIFICATION_MODELS[classification_key]["models"][model_key]
    semaphore  = _get_semaphore(cache_key)
    return _InterpreterContext(model_path, semaphore)


def run_prediction(classification_key: str, model_key: str, img_array: np.ndarray) -> np.ndarray:
    model_path = CLASSIFICATION_MODELS[classification_key]["models"][model_key]
    ctx = get_interpreter(classification_key, model_key)

    if model_path.endswith('.keras'):
        with ctx as c:
            return c.predict(img_array)
    else:
        with ctx as interp:
            input_details  = interp.get_input_details()
            output_details = interp.get_output_details()
            interp.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
            interp.invoke()
            return interp.get_tensor(output_details[0]['index'])


def preprocess_image(file_stream) -> tuple[np.ndarray, str]:
    """Returns (img_array ready for inference, base64-encoded PNG for display)."""
    img = Image.open(file_stream).convert('RGB')

    # Encode original for display
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Resize & normalise
    img_array = img_to_array(img.resize((256, 256))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, image_base64


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        classification    = request.form.get('classification')
        model_architecture = request.form.get('model_architecture')

        if not classification or classification not in CLASSIFICATION_MODELS:
            return render_template('index.html', error_message='Please select a valid classification type.')
        if not model_architecture or model_architecture not in MODEL_ARCHITECTURES:
            return render_template('index.html', error_message='Please select a valid model architecture.')

        classes = CLASSIFICATION_MODELS[classification]["classes"]

        if 'image' not in request.files:
            return render_template('index.html', error_message='No file selected.')

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error_message='No selected file.')

        if not allowed_file(file.filename):
            return render_template('index.html', error_message='Invalid file type. Please upload PNG, JPG, JPEG, or GIF.')

        try:
            img_array, image_base64 = preprocess_image(file.stream)
            preds = run_prediction(classification, model_architecture, img_array)

            predicted_class_index = int(np.argmax(preds))
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

    return render_template('index.html')


@app.route('/error')
def error_page():
    message = request.args.get('message', 'An unknown error occurred.')
    return render_template('error.html', message=message)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)