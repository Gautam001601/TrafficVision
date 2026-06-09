<div align="center">

# 🚗 TrafficVision
### AI-Powered Vehicle Detection & Classification System

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=for-the-badge)
![Django](https://img.shields.io/badge/Django-Backend-green?style=for-the-badge&logo=django)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep Learning-orange?style=for-the-badge&logo=pytorch)

*A full-stack AI web application for real-time vehicle detection and 
classification using a custom-trained YOLOv8 model*

[Live Demo](#) • [Report Bug](#) • [GitHub](https://github.com/Gautam001601/TrafficVision)

</div>

---

## 📌 Overview

**TrafficVision** is an end-to-end AI-powered vehicle detection system 
that identifies and classifies 12 types of vehicles in real-world 
traffic images. Built with YOLOv8 Nano and Django, it features a 
production-ready REST API, Google OAuth authentication, per-user 
detection history, and a Streamlit analytics dashboard.

---

## ✨ Features

- 🔍 **Real-time Vehicle Detection** — Upload any traffic image and get 
  instant detection results with bounding boxes
- 🚘 **12 Vehicle Classes** — big bus, big truck, bus-l, bus-s, car, 
  mid truck, small bus, small truck, truck-l, truck-m, truck-s, truck-xl
- 🔐 **Google OAuth 2.0** — Secure login and signup via Google account
- 📋 **Detection History** — Per-user persistent history with image 
  thumbnails and timestamps
- 📊 **Streamlit Dashboard** — Interactive analytics with metric cards, 
  confidence scores, and bar charts
- ⚡ **GPU Accelerated** — Trained on NVIDIA RTX 5060 with CUDA 12.8

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **ML / DL** | YOLOv8 Nano, PyTorch, Ultralytics |
| **Computer Vision** | OpenCV, Pillow |
| **Backend** | Django, REST API, Django Allauth |
| **Frontend** | HTML, CSS, JavaScript |
| **Dashboard** | Streamlit |
| **Database** | SQLite |
| **Auth** | Google OAuth 2.0 |
| **Training** | CUDA 12.8, NVIDIA RTX 5060 |
| **Data** | Roboflow Universe |

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Precision | 34.3% |
| Recall | 36.0% |
| mAP@0.5 | 16.8% |
| mAP@0.5-0.95 | 9.1% |
| Training Epochs | 50 |
| Dataset Size | 4,058 images |
| Classes | 12 vehicle types |
| Instances | 31,903 annotations |

> **Note:** Lower mAP is attributed to significant class imbalance 
> (car class dominant at 59.8%) and fine-grained similarity between 
> truck sub-categories.

---

## 📁 Project Structure

TrafficVision/
│
├── object/                  # Django Web Application
│   ├── detect/              # Detection app
│   │   ├── models.py        # DetectionLog, Contact models
│   │   ├── views.py         # API & page views
│   │   ├── urls.py          # URL routing
│   │   └── templates/       # HTML templates
│   ├── object/              # Django settings
│   └── manage.py
│
├── app.py                   # Streamlit dashboard
├── train.py                 # YOLOv8 training script
├── evaluate.py              # Model evaluation script
├── config.py                # Project configuration
├── download_dataset.py      # Roboflow dataset downloader
├── requirements.txt
└── README.md

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (optional but recommended)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Gautam001601/TrafficVision.git
cd TrafficVision

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
# Create a .env file and add your keys:
# ROBOFLOW_API_KEY=your_key
# SECRET_KEY=your_django_secret_key

# 5. Run migrations
cd object
python manage.py makemigrations
python manage.py migrate

# 6. Start Django server
python manage.py runserver
```

### Run Streamlit Dashboard
```bash
# From project root
streamlit run app.py
```

---

## 🎯 API Usage

### Detect Vehicles

POST /api/detect/
Content-Type: multipart/form-data
Body: image = <your image file>
Response:
{
"detections": [
{
"class": "car",
"confidence": 0.87,
"bbox": [x1, y1, x2, y2]
}
],
"annotated_image_base64": "..."
}

---

## 📜 Certifications & Training

This project was built during **6-month Industrial Training** at 
**ThinkNEXT Technologies Pvt. Ltd., Mohali** in 
**AI & ML Deep Learning using Python — Grade A**

---

## 👨‍💻 Author

**Gautam Kumar**
- GitHub: [@Gautam001601](https://github.com/Gautam001601)
- University: I.K. Gujral Punjab Technical University, Mohali
- Degree: B.Tech Computer Science & Engineering (2022–2026)

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">
Made with ❤️ by Gautam Kumar
</div>