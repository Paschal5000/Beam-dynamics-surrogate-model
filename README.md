# Development of a Surrogate Model using Artificial Neural Networks for the Dynamic Analysis of Euler-Bernoulli Beams on a Pasternak Foundation

This repository contains the source code and documentation for my final year undergraduate project in Civil Engineering, focused on creating a machine learning-based surrogate model to predict the dynamic response of an Euler-Bernoulli beam on a Pasternak foundation.

## 📖 Project Overview

The dynamic analysis of beams on elastic foundations is a classic problem in structural engineering. Traditional analytical and numerical methods, while accurate, are computationally intensive. This project addresses this challenge by developing a data-driven surrogate model using an Artificial Neural Network (ANN). The model is trained on a high-fidelity dataset generated from the analytical solutions derived by Attama (2025). The goal is to create a tool that can provide instantaneous and accurate predictions of the beam's maximum dynamic deflection, facilitating rapid design iteration and analysis.

### Key Features

  * **Data Generation:** A script to generate a comprehensive dataset from the underlying analytical physics model.
  * **ANN Model:** A deep neural network built with TensorFlow/Keras to learn the complex beam-foundation dynamics.
  * **Model Training:** A complete workflow for pre-processing, training, and validating the surrogate model.
  * **Evaluation:** Rigorous performance evaluation using standard regression metrics.

## 🛠️ Technology Stack

  * **Language:** Python 3.9+
  * **Machine Learning:** TensorFlow (with Keras API)
  * **Data Handling:** Pandas, NumPy
  * **Pre-processing & Metrics:** Scikit-learn
  * **Visualization:** Matplotlib

## 📂 Repository Structure

```
├── data/
│   └── beam_deflection_dataset.csv  # The generated high-fidelity dataset
├── notebooks/
│   └── 1_Data_Exploration.ipynb     # Optional: Jupyter Notebook for exploratory data analysis
├── saved_model/
│   └── surrogate_model.h5           # The final trained and saved model
├── src/
│   ├── generate_dataset.py          # Script to generate the dataset from the analytical model
│   └── train_model.py               # Script for pre-processing, training, and evaluating the model
├── .gitignore                       # Standard Python .gitignore file
├── requirements.txt                 # A list of all required Python libraries
└── README.md                        # This README file
```

## ⚙️ Setup and Usage

To get this project up and running on your local machine, follow these steps.

### 1\. Clone the Repository

```bash
git clone https://github.com/Paschal5000/Beam-dynamics-surrogate-model.git
cd Beam-dynamics-surrogate-model
```

### 2\. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3\. Install Dependencies

Install all the required libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4\. Generate the Dataset

Run the data generation script. This will create the `beam_deflection_dataset.csv` file inside the `data/` directory.

```bash
python src/generate_dataset.py
```

### 5\. Train the Model

Run the training script. This will perform pre-processing, train the ANN, evaluate its performance, and save the final model to the `saved_model/` directory.

```bash
python src/train_model.py
```

## 📊 Preliminary Results

This section will be updated with the final results upon completion of the training and evaluation phase. The primary evaluation metrics will be the **Coefficient of Determination (R² Score)** and **Root Mean Squared Error (RMSE)**.

**Predicted vs. Actual Deflection (Test Set)**
*(A plot will be added here to visually represent the model's accuracy.)*

| Metric | Value |
| :--- | :--- |
| R² Score | (to be filled) |
| RMSE | (to be filled) |
| MAE | (to be filled) |

## 🙏 Acknowledgements

This project is based on the analytical framework and research established in the PhD thesis of **Dr. Attama Chukwuka Malachy**, Department of Civil Engineering, University of Nigeria, Nsukka.

  * Attama, C. M. (2025). *A Rolling Concentrated Load on a Simply Supported Euler Bernoulli Beam on a Pasternak Foundation* [Unpublished doctoral dissertation]. University of Nigeria, Nsukka.
