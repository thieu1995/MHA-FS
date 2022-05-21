#!/usr/bin/env python
# Created by "Thieu" at 00:15, 22/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%


class Config:

    RANDOM_STATE = 42

    DATASET_NAME = "Arrhythmia"
    TEST_SIZE = 0.25
    CLASSIFIER = "RF"               # RF: random forrest, KNN and SVM
    DRAW_CONFUSION_MATRIX = False
    AVERAGE_METRIC = "weighted"     # Used in sklearn.metrics (for multiple-labels classification problem)
    MIN_MAX_PROBLEM = "max"
    OBJ_WEIGHTS = [1, 0, 0, 0]          # Metrics return: [accuracy, precision, recall, f1]
    PRINT_ALL = False

