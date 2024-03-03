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

def test_evaluate_model_with_tolerance(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = train_logistic_regression(X_train, y_train)
    accuracy, precision, recall = evaluate_model(model, X_test, y_test)

    assert accuracy == pytest.approx(0.85, abs=1e-2)  # Check accuracy with a tolerance of 0.01
    assert precision == pytest.approx(0.90, abs=1e-2)  # Check precision with a tolerance of 0.01
    assert recall == pytest.approx(0.80, abs=1e-2)  # Check recall with a tolerance of 0.01

