# Loan Approval Prediction

This project aims to predict loan approval using logistic regression. The dataset is preprocessed, cleaned for outliers, and reduced in dimensionality using PCA. The model's performance is evaluated, and the decision boundary is plotted.

## Project Structure

```
loan_approval/ 
        ├── data/ 
        |   └── loan_approval_dataset.csv 
        ├── output/ 
        |   ├── 1.txt
        |   ├── 2.txt
        |   └── p01b_1.png 
        ├── src/
        |   ├── main.py
        |   └── util.py 
        ├── loan_approval.ipynb 
        └── README.md
```

## Requirements

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/loan_approval.git
    cd loan_approval
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Script

1. Navigate to the [src](http://_vscodecontentref_/1) directory:
    ```bash
    cd src
    ```

2. Run the [main.py](http://_vscodecontentref_/2) script:
    ```bash
    python main.py
    ```

### Jupyter Notebook

Open the Jupyter Notebook:
    ```bash
    jupyter notebook loan_approval.ipynb
    ```

## Project Workflow

1. **Load Data**: The dataset is loaded from a CSV file.
2. **Preprocess Data**: The data is scaled and split into training and test sets.
3. **Train Model**: A logistic regression model is trained on the training set.
4. **Evaluate Model**: The model's accuracy is evaluated on the test set.
5. **Clean Data**: The data is cleaned for outliers based on the model's weights.
6. **PCA**: Principal Component Analysis (PCA) is applied to reduce the dimensionality of the data.
7. **Plot Decision Boundary**: The decision boundary is plotted and saved as an image.

## Results

- The model's accuracy before and after cleaning the data is printed.
- The predictions are saved in the [output](http://_vscodecontentref_/3) directory.
- The decision boundary plot is saved as `output.png` in the [output](http://_vscodecontentref_/4) directory.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
