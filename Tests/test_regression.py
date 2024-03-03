#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pytest
from my_module import train_logistic_regression, evaluate_model
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def test_train_logistic_regression(sample_data):
    X_train, _, y_train, _ = sample_data
    model = train_logistic_regression(X_train, y_train)
    assert hasattr(model, 'coef_')  # Check if the model has been trained and has coefficients

