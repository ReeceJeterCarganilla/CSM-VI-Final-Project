# CSM-VI-Final-Project

1. Project Title
# Aircraft Engine Remaining Useful Life (RUL) Prediction

2. Project Description
## Project Description
This project predicts the Remaining Useful Life (RUL) of aircraft engines using sensor data. Accurate RUL predictions can help schedule maintenance proactively, reduce downtime, and improve safety.

The project uses machine learning models, including Random Forest, Linear Regression, and Decision Trees, to estimate RUL based on operational settings and sensor measurements.

3. Dataset Information
## Dataset
The dataset comes from the NASA Turbofan Engine Degradation Simulation dataset. It includes:
- **Training data**: Simulated engine run-to-failure cycles.
- **Test data**: Partial engine operation data with actual time-to-failure values provided in a separate file.

Each record contains:
- **Engine ID**: A unique identifier for each engine.
- **Cycle**: The operational cycle number.
- **Operational settings**: Three settings that affect engine performance.
- **Sensor data**: Readings from 21 sensors.

[Link to Dataset Documentation](https://data.nasa.gov/).

4. Project Structure
## Project Structure
- `train_FD001.txt`: Training dataset.
- `test_FD001.txt`: Test dataset.
- `RUL_FD001.txt`: Remaining Useful Life for test engines.
- `RUL_Prediction.ipynb`: Jupyter Notebook containing the code for preprocessing, modeling, and evaluation.
- `README.md`: Project documentation.

5. Methodology
## Methodology

1. **Data Preprocessing**:
   - Dropped irrelevant columns.
   - Engine-wise feature engineering (e.g., calculating RUL).
   - Correlation analysis to understand feature relationships.
   - Train-test split.

2. **Modeling**:
   - Implemented and evaluated three models:
     - Random Forest Regressor
     - Linear Regression
     - Decision Tree Regressor
   - Hyperparameter tuning using GridSearchCV.
   - Selected the model with the best performance metrics.

3. **Evaluation**:
   - Used Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) as performance metrics.
   - Compared model predictions with true RUL values.

6. Results
## Results
- The **Random Forest model** outperformed the others, achieving the lowest MAE and RMSE on the test set.
- Model Comparison:
  - **Random Forest**: MAE = `X`, RMSE = `Y`
  - **Linear Regression**: MAE = `A`, RMSE = `B`
  - **Decision Tree**: MAE = `C`, RMSE = `D`

### Visualizations
- [Insert relevant plots, e.g., True vs. Predicted RUL, Feature Importance]

7. How to Run the Project
## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rul-prediction.git
   cd rul-prediction
Install the required Python packages:
pip install -r requirements.txt
Run the Jupyter Notebook:
jupyter notebook RUL_Prediction.ipynb

Follow the steps in the notebook to preprocess the data, train models, and evaluate performance.

### **8. Technologies Used**
```markdown
## Technologies Used
- Python
- Pandas, NumPy, Matplotlib, Seaborn (Data processing and visualization)
- Scikit-learn (Machine learning)
- GridSearchCV (Hyperparameter tuning)
- Jupyter Notebook

9. Future Improvements
## Future Improvements
- Experiment with additional models like Gradient Boosting (e.g., XGBoost, LightGBM).
- Incorporate real-world sensor data for better generalization.
- Implement real-time predictions using a deployed model.
- Explore advanced feature engineering techniques.

11. Contact Information
## Contact
For questions or feedback, feel free to reach out:
- Name: [Your Name]
- Email: [Your Email]
- GitHub: [Your GitHub Profile Link]
