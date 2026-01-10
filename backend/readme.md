🌿 Plant Disease Prediction Website

A web-based application that predicts plant diseases from leaf images using Deep Learning.
The backend is developed with FastAPI, providing a fast and scalable API for model inference.
This project helps farmers and agricultural experts detect plant diseases early and improve crop health.

📌 Project Overview

Plant diseases can significantly reduce crop yield if not identified early.
This application allows users to upload images of plant leaves and receive predictions about possible diseases using a trained Deep Learning (CNN) model.

✨ Key Features

Upload plant leaf images for disease prediction

Deep Learning model for accurate classification

Fast and efficient backend using FastAPI

REST API for easy integration

Swagger UI for API testing

Scalable and production-ready design

🧰 Tech Stack
Backend

Python

FastAPI

Uvicorn

Deep Learning

TensorFlow / Keras

Convolutional Neural Networks (CNN)

NumPy

OpenCV / PIL

Tools

Git & GitHub

Postman (API testing)

🧠 Model Details

CNN-based Deep Learning model

Trained on plant leaf disease images

Image preprocessing: resizing, normalization

Outputs disease name and confidence score

📁 Project Structure
plant-disease-prediction/
│
├── app/
│   ├── main.py          # FastAPI application
│   ├── model.py         # Model loading & prediction
│   ├── utils.py         # Image preprocessing
│
├── model/
│   └── plant_model.h5   # Trained DL model
│
├── requirements.txt
├── README.md

⚙️ How to Run the Project
1. Clone the Repository
git clone https://github.com/your-username/plant-disease-prediction.git
cd plant-disease-prediction

2. Create Virtual Environment
python -m venv venv
source venv/bin/activate
# Windows: venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Start the Server
uvicorn app.main:app --reload


Access the app at:

http://127.0.0.1:8000

🔍 API Usage
Predict Plant Disease

Endpoint: POST /predict

Input:

Image file (leaf image)

Output Example:

{
  "disease": "Tomato Leaf Blight",
  "confidence": 0.92
}

🧪 API Testing

Swagger UI is available at:

http://127.0.0.1:8000/docs

🚀 Future Enhancements

Add frontend UI (React / HTML-CSS)

Deploy using Docker and cloud services

Support more plant species

Improve accuracy using transfer learning

Mobile app integration

📜 License

This project is licensed under the MIT License.

👨‍💻 Author

Akshay Singh
Shikhar Srivastava
Machine Learning / Deep Learning Enthusiast