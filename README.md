# Stroke Risk Predictor Web Application

![GitHub repo size](https://img.shields.io/github/repo-size/your-username/your-repo-name) ![GitHub stars](https://img.shields.io/github/stars/your-username/your-repo-name?style=social) ![GitHub forks](https://img.shields.io/github/forks/your-username/your-repo-name?style=social)
<!-- Optional: Replace 'your-username/your-repo-name' above -->

## Overview

This project is a web application designed to predict the risk of stroke in patients based on various health parameters. It utilizes a machine learning model trained on the "Healthcare Dataset Stroke Data". The application provides a user-friendly interface to input patient details and receive a real-time risk prediction categorized as Low, Medium, or High, along with the calculated probability percentage.

The backend is built with Python using Flask, Scikit-learn, and Pandas, while the frontend uses HTML, CSS, and JavaScript for a responsive user experience.

## Features

*   **Web-Based UI:** Simple and intuitive interface built with HTML, CSS, and JavaScript.
*   **User Input:** Form for users to enter patient details (Age, Gender, Hypertension, Heart Disease, BMI, etc.).
*   **ML Prediction:** Utilizes a trained Logistic Regression model (or your chosen model) to predict stroke probability.
*   **Risk Categorization:** Classifies prediction results into:
    *   **Low Risk** (Displayed in Green)
    *   **Medium Risk** (Displayed in Orange)
    *   **High Risk** (Displayed in Red)
*   **Probability Display:** Shows the calculated stroke probability percentage.
*   **Responsive Design:** Basic responsiveness for usability on different screen sizes.
*   **Background Image:** Features a subtle background image for better aesthetics.

## Screenshots

<!-- Add screenshots of your application here! -->
**Input Form:**
![Input Form Screenshot](link/to/your/screenshot_form.png) <!-- Replace with actual link -->

**Prediction Result (Example: Medium Risk):**
![Prediction Result Screenshot](link/to/your/screenshot_result.png) <!-- Replace with actual link -->

*(Please replace the placeholder links above with actual URLs to your screenshots after uploading them, perhaps to an `assets` folder in your repo)*

## Tech Stack

*   **Backend:**
    *   Python 3.x
    *   Flask (Web Framework)
    *   Scikit-learn (Machine Learning)
    *   Pandas (Data Manipulation)
    *   Joblib (Model Persistence)
*   **Frontend:**
    *   HTML5
    *   CSS3
    *   JavaScript (Fetch API for backend communication)
*   **Dataset:**
    *   `healthcare-dataset-stroke-data.csv` (Source: Specify source, e.g., Kaggle)

## Project Structure

```text
your-repo-name/
├── static/
│   ├── images/
│   │   └── background.jpg      # Background image
│   ├── style.css             # Frontend styling
│   └── script.js             # Frontend logic
├── templates/
│   └── index.html            # Main HTML page
├── .gitignore                # Git ignore file
├── app.py                    # Flask backend application
├── healthcare-dataset-stroke-data.csv # The dataset
├── median_bmi.joblib         # Saved median BMI for imputation
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── stroke_prediction_pipeline.joblib # Saved ML model pipeline
├── train_model.py            # Script to train the model
└── training_columns.joblib   # Saved list of columns used for training
