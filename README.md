AI Student Stress Predictor (MIT209 Final Project)

An interactive, full-stack Machine Learning web application designed to evaluate, log, and predict student stress factors. The application leverages a high-performance **Scikit-Learn** model backend wrapped inside a **Django** web layer, connecting dynamically to a local **MySQL (XAMPP)** database framework for data tracking.


1. Project Goal & Objective

The primary objective of this project is to develop an early-warning predictive system capable of identifying academic and lifestyle indicators that lead to high stress levels in students. By analyzing key daily behavioral signals (such as sleep deprivation, screen time distributions, and perceived anxiety pressures), the system aims to provide data-driven insights to help educators and researchers implement proactive mental health interventions.


2. Dataset Description

The machine learning core is trained on an empirical behavioral dataset tracking student demographics against localized stress triggers. 

**Dataset Source URL:** [Kaggle Student Stress Factors Dataset](https://www.kaggle.com/) *(Replace with your specific dataset URL if applicable)*

Features & Schema Dictionary
The underlying database tracking model (`PredictionHistory`) incorporates these critical behavioral dimensions:

| Column / Feature Name | Data Type | Value Range | Description |
| :--- | :--- | :--- | :--- |
| `student_type` | `IntegerField` | `0` or `1` | Binary categorization: `0` for School Students, `1` for College/University Students. |
| `sleep_hours` | `FloatField` | Continuous | Average duration of sleep acquired by the student per 24-hour cycle. |
| `study_hours` | `FloatField` | Continuous | Average daily duration dedicated purely to academic work and revision. |
| `social_media_hours` | `FloatField` | Continuous | Daily screen-time recorded across modern entertainment and social networks. |
| `attendance` | `FloatField` | `0.0` - `100.0` | Overall percentage of classroom or lecture attendance. |
| `exam_pressure` | `FloatField` | `1.0` - `100.0` | Subjective numerical rating of anxiety felt during evaluation periods. |
| `family_support` | `FloatField` | `1.0` - `10.0` | Perceived stability, mental health encouragement, and domestic backing. |
| `month` | `IntegerField` | `1` - `12` | Calendar tracking dimension to isolate seasonal stress trends (e.g., finals season). |
| `predicted_stress` | `IntegerField` | `0` or `1` | **Target Variable:** The calculated classification output. `0` = Low/Normal Stress, `1` = High Stress. |
| `timestamp` | `DateTimeField` | Automatically Generated | Persistent metadata tracking exactly when the query check occurred. |


3. Data Cleaning & Preparation Techniques

To prepare the structured data for optimal model performance, the following preprocessing workflow was applied:
1. **Handling Missing Values:** Missing features or empty attributes were isolated and handled via median/mean imputation methods to retain distribution shape without skewing variances.
2. **Feature Outlier Removal:** Isolated lifestyle entries with impossible parameters (e.g., $>24$ daily hours) were filtered using Interquartile Range ($IQR$) bounds.
3. **Categorical Encodings:** Nominal values like student academic settings were converted into binary numeric formats (`0`/`1`) via structural mapping to meet Python classification constraints.
4. **Feature Feature Scaling:** Continuous variables like study durations and attendance margins were scaled using `StandardScaler` to ensure features with wider numeric ranges didn't disproportionately distort distance computations.


4. ML Algorithms & Model Performance Metrics

Multiple experimental iterations were explored during training to achieve optimal predictive precision:

**Algorithms Evaluated:**
    **Logistic Regression:** Used as the linear baseline index tracker.
    **Random Forest Classifier:** Implemented as a powerful ensemble learning method to evaluate intricate, non-linear interactions between sleep loss and exam pressure.
    **Support Vector Machine (SVM):** Evaluated to assess boundaries across high-dimensional features.

Final Model Evaluation Metrics
The top-performing model variant saved inside **`best_stress_model.pkl`** achieved robust classification metrics:
**Accuracy:** Quantified structural prediction precision across testing datasets.
**Precision & Recall:** Finetuned to ensure the system reduces False Negatives (missing a student suffering from critically high hidden stress).
**F1-Score:** Outlined harmonic stability across uneven class weights.




### 1. Position Workspace Directory
```powershell
cd C:\xampp\htdocs\DjangoStartMasraMIT209\mit209projMasra
