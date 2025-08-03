# 🖱️ AI Virtual Mouse

A hands-free virtual mouse application built using real-time hand tracking and gesture recognition.

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/3ff70ada-3eec-45c8-9014-89066bf2f9bf" />


## 🚀 Overview

The **AI Virtual Mouse** enables users to control their computer cursor using just hand gestures detected via webcam. This project aims to improve accessibility for users with motor impairments and provide an intuitive human-computer interaction experience.

## 📷 Demo

https://www.linkedin.com/posts/bithal-sahoo-22787a2b6_ai-computervision-machinelearning-activity-7296771352672260096-ZH4n?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEvye78BsX4osw43BRzQiRFDNAQBwZOhaMI



## ✨ Features

- ✅ Real-time hand tracking using webcam
- 👆 Finger gesture-based mouse movement & click control
- 🧠 95%+ gesture detection accuracy
- 💡 No external hardware needed – just a webcam
- 🔁 Smooth and responsive cursor control

## 🛠️ Tech Stack

- **Python**
- **OpenCV** – for real-time image capture & processing
- **MediaPipe** – for accurate hand landmark detection
- **NumPy** – for mathematical operations and array processing

## 📸 How It Works

1. Captures video feed using webcam.
2. Detects hand landmarks using MediaPipe.
3. Maps finger movement to cursor position.
4. Detects gestures (like pinch or finger distance) to perform mouse clicks.


## 📁 Setup Instructions

   
Install Requirements:
pip install -r requirements.txt

Run the Application:
python virtual_mouse.py:

✅ Requirements
Python 3.7+

Webcam (Built-in or External)

📌 Use Cases
Accessibility aid for users with motor impairments

Hands-free PC control in public/sterile environments

Educational computer vision project
