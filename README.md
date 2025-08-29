# CardioPredict - Machine Learning Based Heart Disease Prediction System

## Overview

CardioPredict is a machine learning project designed to predict the likelihood of heart disease in patients using a variety of advanced classification algorithms. The system leverages several state-of-the-art machine learning models and provides a comprehensive comparison using both tabular metrics and visualizations.

---

## Features

- Trains and evaluates multiple ML models on a heart disease dataset
- Compares model performance using accuracy, precision, recall, F1-score, and ROC-AUC
- Predicts heart disease risk for new patient data across all models
- Visualizes results with ROC curves, confusion matrices, and comparison bar charts
- Generates model comparison tables for easy interpretation

---

## Machine Learning Models Used

- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Multi-Layer Perceptron (MLP / Neural Network)**
- **Decision Tree**
- **Random Forest**
- **Gradient Boosting (e.g., XGBoost or GradientBoostingClassifier)**

Each model is trained, evaluated, and compared on the same dataset, enabling robust analysis.

---

## Example Results

### Model Performance Metrics

| Model             | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|-------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.87   | 0.89      | 0.84   | 0.86     | 0.91    |
| KNN                 | 0.85   | 0.86      | 0.83   | 0.84     | 0.88    |
| SVM                 | 0.89   | 0.91      | 0.87   | 0.89     | 0.92    |
| MLP                 | 0.90   | 0.92      | 0.88   | 0.90     | 0.93    |
| Decision Tree       | 0.82   | 0.84      | 0.80   | 0.82     | 0.83    |
| Random Forest       | 0.91   | 0.93      | 0.89   | 0.91     | 0.94    |
| Gradient Boosting   | 0.92   | 0.94      | 0.90   | 0.92     | 0.95    |

> *Note: The above metrics are example results. Actual results may vary depending on dataset splits and hyperparameters.*

---

### Visualizations & Charts

The notebook provides the following charts for analysis:

#### 1. **ROC Curves for All Models**
- Shows the trade-off between sensitivity and specificity for each classifier.
![Image Alt](https://github.com/Debraj-80/CardioPredict-Machine-Learning-Based-Heart-Disease-Prediction-System/blob/741a0e773abbc651eb425bda620db891cf626675/assets/1.png)

#### 2. **Bar Charts for Metric Comparison**
- Compares accuracy, precision, recall, and F1-score across all models.
![Image Alt](https://github.com/Debraj-80/CardioPredict-Machine-Learning-Based-Heart-Disease-Prediction-System/blob/741a0e773abbc651eb425bda620db891cf626675/assets/2.png)
> **Tip:** The notebook auto-generates these charts after training and evaluating the models.

---

### Example Workflow

- Load and inspect the dataset
- Split data into training and testing sets
- Scale features for better model performance
- Train multiple ML models and evaluate each using accuracy, ROC-AUC, confusion matrix, and classification report
- Predict heart disease risk for new patients and visualize the results

---

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook or Google Colab
- Required Python libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - prettytable

### Setup & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Debraj-80/CardioPredict-Machine-Learning-Based-Heart-Disease-Prediction-System.git
   cd CardioPredict-Machine-Learning-Based-Heart-Disease-Prediction-System
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn prettytable
   ```

3. **Run the notebook:**
   - Open `heart_disease_prediction_improved_1.ipynb` in Jupyter Notebook or Google Colab.
   - Follow the instructions in the notebook cells to run the code.

4. **Dataset:**
   - The notebook expects a file named `heart.csv` with heart disease dataset features and target variable.

---

## Important Details

- Each algorithm's predictions are compared and displayed for both model evaluation and new patient prediction.
- Model performance metrics and probability outputs are shown in tables and plots for transparency.
- The system is intended for educational and research purposes and is not suitable for real-world clinical use without further validation.

---

## License

This project currently does not specify a license.

## Author

Developed by [Debraj-80](https://github.com/Debraj-80).

---

