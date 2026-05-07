# 🎓 AI-Based Online Proctoring System

An intelligent AI-powered online proctoring system designed to monitor candidates during online examinations using Computer Vision and Machine Learning techniques. The system helps maintain exam integrity by detecting suspicious activities in real time through webcam analysis.

---

# 📌 Overview

The AI Proctoring System is developed to automate online exam monitoring by using facial detection, eye tracking, object detection, and behavioral analysis. The project aims to reduce malpractice during online assessments and provide a secure examination environment.

The system continuously monitors the user through a webcam and identifies suspicious activities such as:

- Multiple faces appearing on screen
- Candidate absence
- Mobile phone detection
- Frequent head movement
- Looking away from the screen
- Unusual activity detection

---

# 🚀 Features

## ✅ Real-Time Face Detection
Detects the candidate’s face continuously during the examination.

## ✅ Eye Tracking
Tracks eye movement to determine whether the candidate is focused on the screen.

## ✅ Head Pose Detection
Monitors head movement to identify suspicious viewing directions.

## ✅ Multiple Person Detection
Detects if more than one person appears in front of the webcam.

## ✅ Mobile Phone Detection
Identifies the presence of mobile phones during examinations.

## ✅ Suspicious Activity Alerts
Generates alerts whenever suspicious activities are detected.

## ✅ Live Webcam Monitoring
Performs continuous webcam-based monitoring in real time.

## ✅ AI-Based Monitoring
Uses Machine Learning and Computer Vision techniques for intelligent proctoring.

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend Programming |
| OpenCV | Computer Vision |
| TensorFlow | Deep Learning |
| NumPy | Numerical Computation |
| Flask | Web Framework |
| HTML/CSS | Frontend Design |
| JavaScript | Frontend Interaction |
| Machine Learning | Activity Detection |

---

# 📂 Project Structure

```text
ProctoringSystem/
│
├── model/                     # Trained ML models
│
├── static/                    # CSS, JS, Images
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/                 # HTML templates
│   ├── index.html
│   └── dashboard.html
│
├── app.py                     # Main Flask application
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Ignored files
⚙️ Installation Guide
1️⃣ Clone the Repository
git clone https://github.com/kondalreddykishtamgari19/proctoring_system.git
2️⃣ Navigate to Project Directory
cd proctoring_system
3️⃣ Create Virtual Environment
Windows
python -m venv venv
Linux / Mac
python3 -m venv venv
4️⃣ Activate Virtual Environment
Windows
venv\Scripts\activate
Linux / Mac
source venv/bin/activate
5️⃣ Install Required Dependencies
pip install -r requirements.txt
▶️ Running the Project

Start the Flask server using:

python app.py

The application will run on:

http://127.0.0.1:5000/

Open the URL in your browser.
📋 Requirements

Install all dependencies using:

pip install -r requirements.txt

Example dependencies:

Flask
opencv-python
tensorflow
numpy
keras
mediapipe
🧠 Working Principle

The system works in the following steps:

Webcam captures live video feed.
OpenCV processes image frames.
AI/ML models analyze candidate behavior.
Suspicious activities are identified.
Alerts are generated for malpractice detection.
🔍 Detection Capabilities
Detection Type	Description
Face Detection	Detects presence of candidate
Eye Tracking	Monitors eye movement
Head Movement	Detects unusual head rotation
Multiple Face Detection	Detects additional persons
Mobile Detection	Identifies mobile phones
Absence Detection	Detects candidate absence
📸 Screenshots
Home Page

Add screenshot here.

Live Monitoring

Add screenshot here.

Detection Alerts

Add screenshot here.

🔒 Advantages
Reduces exam malpractice
Automated monitoring
Real-time activity analysis
Low human supervision required
AI-powered intelligent detection
🚧 Future Enhancements
Audio Monitoring
Browser Activity Tracking
Cloud Database Integration
Student Authentication System
Exam Recording Storage
Admin Dashboard
AI Behavior Analytics
Multi-user Monitoring
💡 Applications
Online Examinations
Remote Hiring Assessments
Certification Tests
Educational Institutions
Corporate Evaluations
👨‍💻 Author
Kondal Reddy
