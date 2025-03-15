# 📌 XNL Intern Task

## 📖 Overview
The **XNL Intern Task** is a robust testing and monitoring framework designed for machine learning models. It provides tools for model evaluation, drift detection, continuous training, and custom test suites to ensure model reliability and performance over time. This framework is built using Python, Scikit-learn, and custom modules for drift detection and continuous training.

---

## � Framework Preview

<table>
  <tr>
    <th style="width: 50%">📊 Drift Detection</th>
    <th style="width: 50%">🔄 Continuous Training</th>
  </tr>
  <tr>
    <td align="center">
      <img src="https://via.placeholder.com/300" width="250">
    </td>
    <td align="center">
      <img src="https://via.placeholder.com/300" width="250">
    </td>
  </tr>
</table>

---

## 🚀 Features

### 1️⃣ **Model Evaluation**
- Perform **cross-validation** to assess model performance.
- Generate detailed metrics for model evaluation.

### 2️⃣ **Drift Detection**
- Detect **data drift** using statistical tests (Kolmogorov-Smirnov, Wasserstein distance).
- Monitor **concept drift** by tracking model performance over time.

### 3️⃣ **Continuous Training**
- Automatically retrain models when drift is detected.
- Save model versions for reproducibility.

### 4️⃣ **Custom Test Suite**
- Add **edge cases** and **conversation tests** (if applicable).
- Run comprehensive test suites to validate model behavior.

### 5️⃣ **Logging & Monitoring**
- Log all events (training, drift detection, testing) for auditability.
- Generate summaries for drift, training, and test results.

---

## 🛠️ **Tech Stack**
- **Python**: Core programming language.
- **Scikit-learn**: Machine learning model training and evaluation.
- **NumPy**: Numerical computations and data generation.
- **Pandas**: Data manipulation and analysis.
- **Joblib**: Model serialization and storage.
- **Logging**: Event logging and monitoring.

---

## 📂 **Project Structure**
plaintext
```
📦 xnl-intern-task
 ┣ 📜 README.md                 # Project documentation
 ┣ 📜 main.py                   # Main script to run the framework
 ┣ 📜 requirements.txt          # Dependencies
 ┣ 📂 testing                   # Testing and monitoring modules
 ┃ ┣ 📂 model_evaluation        # Model evaluation scripts
 ┃ ┃ ┗ 📜 evaluator.py         # Model evaluator module
 ┃ ┣ 📂 monitoring              # Monitoring scripts
 ┃ ┃ ┣ 📜 drift_detector.py    # Drift detection module
 ┃ ┃ ┗ 📜 continuous_trainer.py# Continuous training module
 ┃ ┣ 📂 test_suites            # Custom test suites
 ┃ ┃ ┗ 📜 custom_test_suite.py # Custom test suite module
 ┣ 📂 logs                     # Log files
 ┣ 📂 models                   # Saved model versions
 ┣ 📂 results                  # Test and evaluation results

---
```
## 📦 **Installation & Setup**
1️⃣ **Clone the repository:**
```sh
git clone https://github.com/your-username/react-native-todo-list-app.git
cd react-native-todo-list-app
```

2️⃣ **Install dependencies:**
```sh
pip install -r requirements.txt

```

3️⃣ **Start the Expo development server:**
```sh
python main.py
```
---

Happy Coding! 🚀

