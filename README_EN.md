
---

# **README.md (English)**

```markdown
# Fraud Detection in Transactions

## Project Description
The goal of this project is to detect fraudulent transactions using Machine Learning. This helps identify suspicious behavior and reduce financial risk for the company.

## Dataset
This project uses the **Credit Card Fraud Detection** dataset from Kaggle.

- Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Description: Anonymized credit card transactions labeled as fraudulent or legitimate.
Due to its size, the dataset is **not included in this repository**.  
To reproduce the results, download the dataset from Kaggle and place it in the `data/` directory.
- Features: V1 to V28 (anonymized), Amount, Class
- Target variable: `Class` (0 = Not fraud, 1 = Fraud)

## Project Structure

fraud_detection/
├── data/             # Original and cleaned dataset for testing
├── src/              # Source code: loading, preprocessing, modeling, evaluation
├── notebooks/        # Exploratory data analysis (EDA) and modeling
├── models/           # Trained models and scaler
├── results/          # Generated plots and metrics
├── main.py           # Main script to run the full pipeline
├── README.md         # This file
├── requirements.txt  # Project dependencies

## Data Preprocessing
- Drop missing values  
- Scale numeric features (StandardScaler)  
- Separate features and target (`Class`)  
- Train-test split

## Trained Models
- **Logistic Regression**: linear and interpretable  
- **Random Forest**: robust, handles non-linear relationships  
- **Linear SVM**: efficient with scaled data

## Evaluation Metrics
- Accuracy  
- Precision, Recall, F1-score  
- ROC-AUC  
- Confusion Matrix

## Key Conclusions
- Random Forest is the most robust model on this dataset  
- Most important features according to Random Forest: V11, V4, V2, V21  
- Data cleaning and scaling are critical for model performance

## Recommendations
- Monitor suspicious transactions automatically  
- Review alerts based on the most relevant features  
- Continue collecting data to improve models

## How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python main.py
