# ❤️ Heart Disease Prediction Using Machine Learning

## 📌 Project Overview

This project predicts whether a patient is likely to have heart disease based on various medical parameters using Machine Learning.

The model was built using K-Nearest Neighbors (KNN) and deployed using Streamlit for real-time predictions.

---

## 🚀 Features

- Interactive Streamlit Web Application
- Real-Time Heart Disease Prediction
- Data Preprocessing & Feature Scaling
- Hyperparameter Tuned KNN Model
- User-Friendly Interface
- Instant Prediction Results

---

## 📊 Dataset Features

The dataset contains the following medical parameters:

| Feature | Description |
|----------|------------|
| age | Age of patient |
| sex | Gender |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| fbs | Fasting blood sugar |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression |
| slope | Slope of ST segment |
| ca | Number of major vessels |
| thal | Thalassemia |

### Target Variable

- 0 → No Heart Disease
- 1 → Heart Disease Detected

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Plotly

---

## 📈 Model Performance

### K-Nearest Neighbors (KNN)

| Model | Accuracy |
|---------|----------|
| KNN Before Tuning | 86.34% |
| KNN After Hyperparameter Tuning | 100.00% |

### Note

The tuned KNN model achieved excellent performance on the test dataset. Additional validation techniques such as cross-validation can be applied to verify generalization performance.

---

## 📂 Project Structure

```text
HeartDisease_PredictionKNN/
│
├── app.py # Streamlit application
├── model.pkl # Trained KNN model
├── scaler.pkl # Feature scaler
├── requirements.txt # Dependencies
├── README.md # Project documentation
│
├── images/
│ ├── dashboard.png
│ ├── prediction1.png
│ ├── prediction2.png
```

---

## 🖥️ Application Preview

### Dashboard

![Dashboard](images/dashboard.png)

### Prediction Result

![Prediction1](images/prediction1.png)

### Prediction Result

![Prediction2](images/prediction2.png)

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/Akhilajan27/Machine-Learning-Projects.git
```

### Navigate to Project Folder

```bash
cd Machine-Learning-Projects/HeartDisease_PredictionKNN
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run deploy.py
```

---

## 🔄 Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Selection
5. Data Scaling
6. Model Training
7. Hyperparameter Tuning
8. Model Evaluation
9. Streamlit Deployment

---

## 🔮 Future Improvements

- Cross Validation
- Model Explainability
- Cloud Deployment
- Enhanced Dashboard UI
- Multiple Model Comparison

---

## 👩‍💻 Author

**Akhila**

Aspiring Data Scientist | Machine Learning Enthusiast

### Connect With Me

- GitHub: https://github.com/Akhilajan27
- LinkedIn: www.linkedin.com/in/akhila-k-tech

---

## ⭐ Support

If you found this project useful, please give it a star ⭐ on GitHub.