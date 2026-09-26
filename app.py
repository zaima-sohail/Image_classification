import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model


IMAGE_SIZE = (180, 180)
MODEL_PATH = Path(__file__).resolve().parent / "Image_classify.keras"
CLASS_NAMES = [
	"apple",
	"banana",
	"beetroot",
	"bell pepper",
	"cabbage",
	"capsicum",
	"carrot",
	"cauliflower",
	"chilli pepper",
	"corn",
	"cucumber",
	"eggplant",
	"garlic",
	"ginger",
	"grapes",
	"jalepeno",
	"kiwi",
	"lemon",
	"lettuce",
	"mango",
	"onion",
	"orange",
	"paprika",
	"pear",
	"peas",
	"pineapple",
	"pomegranate",
	"potato",
	"raddish",
	"soy beans",
	"spinach",
	"sweetcorn",
	"sweetpotato",
	"tomato",
	"turnip",
	"watermelon",
]


@st.cache_resource(show_spinner="Loading classifier...")
def get_model(model_digest: str, _model_bytes: bytes) -> tf.keras.Model:
	with tempfile.TemporaryDirectory() as temporary_directory:
		model_path = Path(temporary_directory) / "classifier.keras"
		model_path.write_bytes(_model_bytes)
		with zipfile.ZipFile(model_path) as archive:
			model_config = json.loads(archive.read("config.json"))

		if _remove_null_quantization_config(model_config):
			compatible_model_path = Path(temporary_directory) / "compatible.keras"
			with (
				zipfile.ZipFile(model_path) as source,
				zipfile.ZipFile(compatible_model_path, "w", zipfile.ZIP_DEFLATED) as target,
			):
				for member in source.infolist():
					content = source.read(member.filename)
					if member.filename == "config.json":
						content = json.dumps(model_config).encode("utf-8")
					target.writestr(member, content)
			model_path = compatible_model_path

		return load_model(model_path, compile=False)


def _remove_null_quantization_config(value: object) -> bool:
	changed = False
	if isinstance(value, dict):
		if value.get("quantization_config", object()) is None:
			del value["quantization_config"]
			changed = True
		for nested_value in value.values():
			changed = _remove_null_quantization_config(nested_value) or changed
	elif isinstance(value, list):
		for nested_value in value:
			changed = _remove_null_quantization_config(nested_value) or changed
	return changed


def predict_image(model: tf.keras.Model, image: Image.Image) -> np.ndarray:
	resized_image = image.convert("RGB").resize(IMAGE_SIZE)
	image_array = np.asarray(resized_image, dtype=np.float32)
	predictions = model.predict(np.expand_dims(image_array, axis=0), verbose=0)
	probabilities = np.asarray(predictions)[0]

	if probabilities.ndim != 1 or probabilities.size != len(CLASS_NAMES):
		raise ValueError(
			"The model output does not match the 36 fruit and vegetable classes."
		)
	if not np.isfinite(probabilities).all():
		raise ValueError("The model returned invalid prediction values.")

	return probabilities


st.set_page_config(page_title="Produce Classifier", page_icon="🥝", layout="centered")
st.title("Produce Classifier")
st.caption("Fruit and vegetable recognition")

uploaded_file = st.file_uploader(
	"Choose an image",
	type=("jpg", "jpeg", "png", "webp", "bmp"),
	key="image_upload",
)

image = None
if uploaded_file is not None:
	try:
		image = Image.open(uploaded_file).convert("RGB")
	except (OSError, ValueError):
		st.error("This file could not be read as an image.")
		st.stop()

	st.image(image, caption=uploaded_file.name, use_container_width=True)

uploaded_model = st.file_uploader(
	"Optional Keras model override",
	type=("keras",),
	key="model_upload",
)

if uploaded_model is not None:
	model_bytes = uploaded_model.getvalue()
	model_name = uploaded_model.name
elif MODEL_PATH.is_file():
	model_bytes = MODEL_PATH.read_bytes()
	model_name = MODEL_PATH.name
else:
	st.error(f"Model file not found: {MODEL_PATH.name}")
	st.info("Place the trained model beside `app.py` or upload it above.")
	st.stop()

if image is None:
	st.info("Choose an image to classify.")
	st.stop()

model_digest = hashlib.sha256(model_bytes).hexdigest()
st.caption(f"Model: {model_name}")

try:
	classifier = get_model(model_digest, model_bytes)
except Exception as error:
	st.error("The uploaded model could not be loaded. Choose a valid `.keras` model.")
	st.exception(error)
	st.stop()

try:
	with st.spinner("Classifying image..."):
		probabilities = predict_image(classifier, image)
except (ValueError, tf.errors.OpError) as error:
	st.error(f"Could not classify this image: {error}")
	st.stop()

best_index = int(np.argmax(probabilities))
st.subheader(CLASS_NAMES[best_index].title())
st.metric("Confidence", f"{probabilities[best_index] * 100:.1f}%")

top_indices = np.argsort(probabilities)[-5:][::-1]
st.markdown("**Top predictions**")
for index in top_indices:
	confidence = float(np.clip(probabilities[index], 0.0, 1.0))
	st.write(f"{CLASS_NAMES[index].title()}  ·  {confidence * 100:.1f}%")
	st.progress(confidence)