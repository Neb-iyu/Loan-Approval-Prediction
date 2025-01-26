import pandas as pd
import numpy as np
import util
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from pandas.plotting import parallel_coordinates
def main():
    #Load data
    x, y = util.load_data('../data/loan_approval_dataset.csv', add_intercept=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state=42)
    
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = np.mean(np.round(y_pred) == y_test)
    print(f'Accuracy: {accuracy}')
    print(f'Weights: {model.theta}')

    #Clean the data for outliers
    x = clean(x, model.theta)
    #Split the data again and train the model
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = np.mean(np.round(y_pred) == y_test)
    theta = model.theta.copy()
    print(f'Accuracy after cleaning: {accuracy}')
    print(f'Weights after cleaning: {model.theta}')
    
    np.savetxt('../output/1.txt', y_pred > 0.5, fmt='%d')
    np.savetxt('../output/2.txt', y_test > 0.5, fmt='%d')
    #Reduce the dimensionality of the data using PCA
    pca = PCA(n_components=2)
    x_train_pca = pca.fit_transform(x_train)
    x_train_pca = util.add_intercept(x_train_pca)
    x_test_pca = pca.transform(x_test)
    x_test_pca = util.add_intercept(x_test_pca)
    
    #Train the model on the reduced dimensionality data
    model.fit(x_train_pca, y_train)
    y_pred = model.predict(x_test_pca)
    accuracy = np.mean(np.round(y_pred) == y_test)
    print(f'Weights after PCA: {model.theta}')
    print(f'Accuracy after PCA: {accuracy}')
    print(theta)
    
    util.plot(x_test, y_test, theta, save_path='../output/output.png')

class LogisticRegression:

    def fit(self, x, y):
        '''
        Logistic model using Newthon's Method
        Args:
            X: Numpy array of training inputs. Shape(m, n)
            y: Numpy array of training labels. Shape(m,)
        '''
        m, n = x.shape
        self.theta = np.zeros(n)
        while True:
            theta_old = np.copy(self.theta)
            h = np.divide(1, (1 + np.exp(-(np.matmul(x, self.theta)))))
            gradient = (-1/m) * np.matmul(x.T, (y - h))
            h = np.reshape(h, (-1,1))
            hessian = (1/m) * np.dot(x.T, h * (1-h) * x)
            self.theta -= np.dot(np.linalg.inv(hessian), gradient)
            if np.linalg.norm(self.theta-theta_old, ord=1) < 1e-5:
                break
    
    def predict(self, x):
        """
        Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        return 1 / (1 + np.exp(-np.matmul(x, self.theta)))

def clean(X, theta):
    '''
    Clean the data by removing the outliers
    Args:
        X: Numpy array of shape (n, d)
        theta: Numpy array of shape (d+1,)
    
    Returns:
        X: Numpy array of shape (n, d-j) where j is the number of columns removed
    '''
    to = [i for i in range(len(theta)) if np.abs(theta[i]) < 1e-6]
    print(f'Columns removed: {to}')
    return np.delete(X, to, axis=1)


if __name__ == '__main__':
    main()