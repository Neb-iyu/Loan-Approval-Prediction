import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def add_intercept(X):
    '''
    Add a column of ones to the input matrix X

    Args:
        X: Numpy array of shape (n, d)

    Returns:
        X: Numpy array of shape (n, d+1)
    '''
    return np.column_stack((np.ones(X.shape[0]), X))

#Read the data
def load_data(path, add_intercept=False):
    '''
    Load data from a csv file and return X and y
    Args:
        path: Path to csv file containing dataset
        add_intercept: Add a column of ones to X
    Returns:
        X: Numpy array of x values (inputs)
        y: Numpy array of y values (labels)
    '''

    def addIntercept(X):
        global add_intercept
        return add_intercept(X)
    
    dict = {' Approved':1, ' Rejected':0,
            ' Yes':1, ' No':0,
            ' Graduate':1, ' Not Graduate':0}
    with open(path, 'r') as f:
        header = f.readline().strip().split(',')
    
    df = pd.read_csv(path).replace(dict)
    X = df.iloc[:, 1:-1].replace(dict).values
    y = df[' loan_status'].values
    if add_intercept:
        X = addIntercept(X)
    
    return X, y


def plot(x, y, theta, save_path=None):

    plt.figure()
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)

    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    # Calculate margins for both dimensions, allowing larger margins
    margin_x1 = (np.max(x[:, -2]) - np.min(x[:, -2])) 
    margin_x2 = (np.max(x[:, -1]) - np.min(x[:, -1]))

    # Determine plotting range (with margin applied)
    x1_min, x1_max = np.min(x[:, -2]) - margin_x1, np.max(x[:, -2]) + margin_x1
    x2_min, x2_max = np.min(x[:, -1]) - margin_x2, np.max(x[:, -1]) + margin_x2

    # Create a range of x1 values for decision boundary
    x1_vals = np.linspace(x1_min, x1_max, 100)

    # Compute the decision boundary x2 values using theta
    x2_vals = -(theta[0] + theta[-2] * x1_vals) / theta[-1]

    # Plot decision boundary
    plt.plot(x1_vals, x2_vals, c='red', linewidth=2)

    # Set plot limits based on the calculated range with margins
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)

    # Add labels and title
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Logistic Regression Decision Boundary')

    # Save the plot if save_path is provided
    if save_path is not None:
        plt.savefig(save_path)
    
    # Show the plot
    plt.show()