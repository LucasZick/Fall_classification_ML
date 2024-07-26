# Machine Learning Model Evaluation

This project involves analyzing different machine learning models for classifying falls and activities of daily living (ADL). The process is divided into three main steps:

1. **Dataset Creation** (`create_dataset.py`)
2. **Data Processing and Model Training** (`process.py`)
3. **Model Accuracy Visualization** (`plot_accurs.py`)

## Project Structure

- `create_dataset.py`: This script extracts features from raw data and creates training and test datasets.
- `process.py`: This script trains various machine learning models and evaluates their accuracies, saving the results to a CSV file.
- `plot_accurs.py`: This script loads the saved model accuracies and plots a bar chart showing the comparison between models.

## Usage Instructions

### Step 1: Dataset Creation

Run the `create_dataset.py` script to create the dataset with extracted features and prepare the training and test sets.

`````
python create_dataset.py
`````

### Step 2: Data Processing and Model Training

After creating the dataset, run the `process.py` script to train the models and evaluate their accuracies. This script will save the results to a CSV file.

`````
python process.py
`````

### Step 3: Model Accuracy Visualization

Finally, run the `plot_accurs.py` script to visualize the accuracy of the models in a bar chart. The chart will be saved as a PNG image and displayed on the screen.

`````
python plot_accurs.py
`````

## Script Descriptions

### `create_dataset.py`

This script performs the following operations:

1. **Data Loading**: Loads raw data from the `Fall` and `ADL` folders.
2. **Feature Extraction**: Extracts relevant features from the data.
3. **Dataset Creation**: Saves the training and test datasets to CSV files.

### `process.py`

This script performs the following operations:

1. **Data Loading**: Loads the training and test datasets.
2. **Model Training**: Trains various machine learning models, including Perceptron, ADALINE, Logistic Regression, SVM, Decision Trees, Random Forests, k-Nearest Neighbors, AdaBoost, Gradient Boosting, XGBoost, and LightGBM.
3. **Accuracy Evaluation**: Evaluates the models and saves the accuracies to a CSV file.

### `plot_accurs.py`

This script performs the following operations:

1. **Accuracy Loading**: Loads model accuracies from the CSV file.
2. **Chart Generation**: Plots a bar chart showing the accuracy of each model and saves the image as a PNG file.

## Dependencies

Ensure you have the following libraries installed:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `lightgbm`

You can install these libraries using pip:

`````
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm
`````

## Contribution

Feel free to contribute improvements, bug fixes, or suggestions. To contribute, please submit a pull request or open an issue in the repository.
