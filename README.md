H1N1 Vaccination Prediction | ALGA Hackathon

Dataset B · Machine Learning Classification

This repository contains a machine learning workflow developed for the ALGA Hackathon H1N1 Challenge. The goal is to predict whether a respondent received the H1N1 vaccine using survey, health, behavioural, and demographic information.

The project covers exploratory data analysis, missing-value handling, model benchmarking, hyperparameter tuning, holdout evaluation, feature importance, threshold analysis, and generation of competition submission files.

Project overview

|Item                          |Details                                                                                           |
|------------------------------|--------------------------------------------------------------------------------------------------|
|Task                          |Binary classification of H1N1 vaccination status                                                  |
|Target                        |h1n1_vaccine (0 = not vaccinated; 1 = vaccinated)                                               |
|Identifier                    |respondent_id                                                                                   |
|Features                      |29                                                                                                |
|Training samples              |4,756                                                                                             |
|Test samples                  |4,749                                                                                             |
|Validation strategy           |Stratified 80/20 development–holdout split; 5-fold stratified cross-validation on development data|
|Random seed                   |42                                                                                              |
|Primary model selection metric|Cross-validation ROC-AUC                                                                          |

Models evaluated

The notebook compares five classifiers and a majority-class baseline:

* Logistic Regression
* Random Forest
* Histogram-based Gradient Boosting
* XGBoost
* CatBoost
* Majority-class dummy baseline

Each classifier is wrapped in a scikit-learn preprocessing pipeline. Numerical features use median imputation with missingness indicators and standard scaling; categorical features use constant-value imputation and one-hot encoding. Hyperparameters are tuned with RandomizedSearchCV using six sampled configurations per model and stratified 5-fold cross-validation.

Results

Tuned model performance on the holdout set

The notebook reports the following metrics on a separate 952-row holdout set, with a classification threshold of 0.5:

|Model                      |ROC-AUC   |Accuracy  |F1-score  |
|---------------------------|---------:|---------:|---------:|
|*XGBoost*                |*0.8457|0.7826|0.7137*|
|CatBoost                   |0.8450    |0.7826    |0.7047    |
|Histogram Gradient Boosting|0.8411    |0.7763    |0.7004    |
|Logistic Regression        |0.8380    |0.7784    |0.6973    |
|Random Forest              |0.8332    |0.7679    |0.6687    |

XGBoost has the highest holdout ROC-AUC. However, CatBoost has the highest tuned development cross-validation ROC-AUC (approximately 0.8577) and is therefore ranked first for the notebook’s submission generation. This distinction preserves the holdout set as an evaluation set rather than using it for model selection.

Feature importance

Permutation importance on the holdout set identifies several influential predictors, including:

1. doctor_recc_h1n1 — doctor’s H1N1 vaccine recommendation
2. health_insurance — health insurance status
3. opinion_h1n1_vacc_effective — belief in vaccine effectiveness
4. opinion_h1n1_risk — perceived H1N1 risk
5. health_worker — healthcare worker status

These are predictive associations, not evidence of causality.

Workflow

1. Load and validate data: Check target labels, identifiers, columns, and train/test consistency.
2. Explore the data: Examine class balance, missing values, feature relationships, and vaccination rates.
3. Prepare training data: Create a stratified development/holdout split and preprocessing pipelines.
4. Benchmark models: Compare classifiers with stratified cross-validation against a dummy baseline.
5. Tune hyperparameters: Optimize model configurations using cross-validated ROC-AUC.
6. Evaluate on holdout: Report ROC-AUC, accuracy, precision, recall, F1, log loss, Brier score, confusion matrices, ROC curves, and precision–recall curves.
7. Analyze decisions and explanations: Inspect classification thresholds, cost-related scenarios, and permutation feature importance.
8. Generate submissions: Refit tuned models on the full training data, predict the test set, and export the outputs.

Repository structure

.
├── ALGA_Hackathon_H1N1_Challenge.ipynb   # End-to-end analysis and modeling
├── README.md                             # Project documentation
└── data/                                 # Add locally if the dataset can be shared
    ├── dataset_B_training.csv
    ├── dataset_B_testing.csv
    └── dataset_B_example_submission.csv

The data/ directory above is a suggested layout, not an automatically created folder. The CSV files were not included with this README.

Getting started

1. Requirements

Use Python 3 with Jupyter Notebook or Google Colab. Install the main dependencies:

pip install numpy pandas matplotlib seaborn scikit-learn xgboost catboost shap notebook

shap is installed by the notebook, although its analysis primarily uses permutation importance.

2. Provide the dataset

Obtain the three Dataset B CSV files from the challenge organisers:

* dataset_B_training.csv
* dataset_B_testing.csv
* dataset_B_example_submission.csv

Important: The notebook currently defines input paths pointing to a specific Google Drive directory:

TRAIN_FILE = "/content/drive/MyDrive/Herts/Hackathon/dataset_B_training.csv"
TEST_FILE = "/content/drive/MyDrive/Herts/Hackathon/dataset_B_testing.csv"
EXAMPLE_FILE = "/content/drive/MyDrive/Herts/Hackathon/dataset_B_example_submission.csv"

Update these paths to match your environment (for example, data/dataset_B_training.csv, etc.) before running the data-loading cell. The notebook includes a Colab upload prompt when the original paths are not found, but the paths must still be updated to refer to the uploaded files.

3. Run the notebook

Open ALGA_Hackathon_H1N1_Challenge.ipynb in Jupyter or Google Colab and execute the cells in order. The notebook installs or upgrades XGBoost, CatBoost, and SHAP in its first code cell.

Generated outputs

Running the full notebook generates the following files in the current working directory:

|Output                                  |Description                                           |
|----------------------------------------|------------------------------------------------------|
|cross_validation_results.csv          |Initial model comparison across CV folds              |
|tuning_results.csv                    |Best tuned parameters and CV ROC-AUC per model        |
|holdout_model_comparison.csv          |Holdout performance across evaluated models           |
|permutation_importance.csv            |Permutation importance scores                         |
|submission_manifest.csv               |Ranking, filenames, and metadata for submission files |
|1_dataset_B_catboost.csv              |First-ranked submission based on development CV       |
|2_dataset_B_hist_gradient_boosting.csv|Second-ranked submission                              |
|3_dataset_B_xgboost.csv               |Third-ranked submission                               |
|4_dataset_B_random_forest.csv         |Fourth-ranked submission                              |
|5_dataset_B_logistic_regression.csv   |Fifth-ranked submission                               |
|H1N1_Dataset_B_outputs.zip            |Archive of generated submissions and CSV result tables|

