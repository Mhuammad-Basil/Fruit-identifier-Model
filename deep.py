import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow import keras
model = keras.models.load_model("fruit_classifier_model.keras")
class_names = sorted([
    folder for folder in os.listdir("fruit_images/Training")
    if not folder.startswith(".")
])
def predict_image(img_path):
    img = Image.open(img_path).resize((100, 100))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img_resized = img.resize((200, 200))
        tk_img = ImageTk.PhotoImage(img_resized)
        image_label.config(image=tk_img)
        image_label.image = tk_img
        result = predict_image(file_path)
        result_label.config(text=f"Prediction: {result}")
root = tk.Tk()
root.title("Fruit Classifier")
root.geometry("300x400")
upload_button = tk.Button(root, text="Upload Fruit Image", command=upload_image)
upload_button.pack(pady=10)
image_label = tk.Label(root)
image_label.pack()
result_label = tk.Label(root, text="Visualization", font=("Arial",12))
result_label.pack(pady=20)
root.mainloop()
