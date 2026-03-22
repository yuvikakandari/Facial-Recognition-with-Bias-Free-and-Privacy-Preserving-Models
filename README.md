Face Recognition with Bias Analysis & Privacy Features
🚀 Overview

This project implements a real-time facial recognition system using Deep Learning (DeepFace) and evaluates system fairness using FAR, FRR, and ROC curves.

✨ Features

Face Recognition using DeepFace embeddings

Real-time webcam detection

Unknown face blurring (privacy feature)

Bias evaluation using FAR & FRR

ROC Curve visualization

🧠 Tech Stack

Python

OpenCV

DeepFace

Scikit-learn

Matplotlib

📊 Evaluation Metrics

False Acceptance Rate (FAR)

False Rejection Rate (FRR)

ROC Curve

🔒 Privacy Features

Unknown faces are blurred

No data stored externally

📂 Folder Structure
dataset/
recognition_deepface.py
metrics.py
roc_curve.py
main.py
▶️ How to Run
pip install -r requirements.txt
python main.py
📈 Future Improvements

Federated Learning

Differential Privacy

Streamlit Web App