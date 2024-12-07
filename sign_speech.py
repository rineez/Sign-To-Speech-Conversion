import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import os
import numpy as np
from keras.models import load_model  # For gesture recognition
from PIL import Image, ImageTk
from gtts import gTTS
import pyttsx3

# Load pre-trained gesture recognition model (you need a model trained for sign language)
# MODEL_PATH = "gesture_model.h5"  # Replace with your model path
# gesture_model = load_model(MODEL_PATH)

# Gesture labels corresponding to the model's output
gesture_labels = {0: "Hello", 1: "Thanks", 2: "Yes", 3: "No", 4: "Help"}  # Add more labels as needed

# Pyttsx3 engine for text-to-speech support
engine = pyttsx3.init()

# Function to capture webcam frames and process gestures
def recognize_sign_language():
    cap = cv2.VideoCapture(0)
    recognized_text = ""

    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to access webcam.")
        return recognized_text

    messagebox.showinfo("Info", "Press 'q' to stop recognition.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame for gesture model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(frame_rgb, (224, 224))  # Assuming 224x224 input size for the model
        input_data = np.expand_dims(resized_frame / 255.0, axis=0)

        # Predict gesture
        prediction = 0 # gesture_model.predict(input_data)
        gesture_index = np.argmax(prediction)
        recognized_text = gesture_labels.get(gesture_index, "Unknown Gesture")

        # Display frame with the recognized gesture
        cv2.putText(frame, recognized_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Sign Language Recognition (Press 'q' to exit)", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return recognized_text

# Function to convert text to speech
def convert_text_to_audio(text, language):
    try:
        if language == "ml":  # Use pyttsx3 for Malayalam
            engine.setProperty("voice", "com.apple.speech.synthesis.voice.malayalam")
            engine.say(text)
            engine.runAndWait()
        else:  # Use gTTS for other languages
            tts = gTTS(text, lang=language)
            filename = "output_audio.mp3"
            tts.save(filename)
            os.system(f"start {filename}")
    except Exception as e:
        messagebox.showerror("Error", f"Audio generation failed: {e}")

# Function to handle sign language conversion
def handle_sign_language_conversion():
    recognized_text = recognize_sign_language()
    if recognized_text:
        input_text.delete("1.0", "end")
        input_text.insert("1.0", recognized_text)
        selected_language = language_var.get()

        if not selected_language:
            messagebox.showwarning("Language Error", "Please select a language.")
            return

        convert_text_to_audio(recognized_text, selected_language)
        messagebox.showinfo("Success", f"Recognized Gesture: {recognized_text}")
    else:
        messagebox.showwarning("Error", "No gesture recognized!")

# Main Tkinter window
root = tk.Tk()
root.title("Sign Language to Audio Converter")

# Language options
languages = {
    "Hindi": "hi",
    "Kannada": "kn",
    "French": "fr",
    "English": "en",
    "Arabic": "ar",
    "Greek": "el",
    "Gujarati": "gu",
    "Malayalam": "ml",
    "Punjabi": "pa"
}

# UI Elements
frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky="nsew")

# Text input
ttk.Label(frame, text="Recognized Text:").grid(row=0, column=0, sticky="w")
input_text = tk.Text(frame, wrap="word", width=50, height=10)
input_text.grid(row=1, column=0, columnspan=2, pady=10)

# Language selection dropdown
ttk.Label(frame, text="Choose Language:").grid(row=2, column=0, sticky="w")
language_var = tk.StringVar()
language_dropdown = ttk.Combobox(frame, textvariable=language_var, state="readonly", width=30)
language_dropdown['values'] = list(languages.keys())
language_dropdown.grid(row=2, column=1, sticky="e")

# Buttons
sign_language_button = ttk.Button(frame, text="Convert Sign Language to Audio", command=handle_sign_language_conversion)
sign_language_button.grid(row=3, column=0, columnspan=2, pady=5)

# Run Tkinter event loop
root.mainloop()