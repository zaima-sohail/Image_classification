# Produce Classifier

A Streamlit app that classifies uploaded fruit and vegetable images with a TensorFlow/Keras model.

<img width="1917" height="971" alt="image" src="https://github.com/user-attachments/assets/3c44d73c-35a7-4a59-88fe-a709a47abf94" />
<img width="1416" height="665" alt="image" src="https://github.com/user-attachments/assets/9ed23f97-adb6-4d4f-a7c1-53f36a105eec" />


## Project Files

- `app.py` - Streamlit image upload, preprocessing, and prediction interface.
- `Image_classify.keras` - Trained 36-class classifier used by default.
- `Image Classification Project.ipynb` - Dataset exploration and model training notebook.
- `requirements.txt` - Python dependencies.

## Setup on Windows

Use Python 3.10. The commands below create a separate environment without changing the existing `.venv`:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe" -m venv .venv310
.\.venv310\Scripts\python.exe -m pip install --upgrade pip
.\.venv310\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the App

```powershell
.\.venv310\Scripts\python.exe -m streamlit run app.py
```

Streamlit prints the local URL after the server starts, usually `http://localhost:8501`.

## Classify an Image

1. Open the Streamlit page and upload a JPG, JPEG, PNG, WebP, or BMP image.
2. The app loads `Image_classify.keras` from the same folder as `app.py` automatically.
3. An optional `.keras` upload can override the local model.
4. The app displays the top prediction, confidence, and five highest-scoring classes.

The model expects 180 x 180 RGB images with pixel values in the original 0-255 range. A replacement model must return probabilities for the same 36 classes in the order defined in `app.py`.

## Training Notebook

The notebook downloads the Fruit and Vegetable Image Recognition dataset through KaggleHub and trains a CNN. The dataset download requires an internet connection. The notebook currently does not save the trained model; the Streamlit app requires a `.keras` model file, so a model produced by retraining must be saved or uploaded separately before it can be used.
