#!/usr/bin/env python
# Created by "Thieu" at 00:14, 22/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.evolutionary_based import DE
from sklearn.model_selection import train_test_split
import numpy as np
from src.config import Config
from src.utils.data_util import get_dataset
from src.utils.metric_util import Evaluator


def amend_position(position, lower, upper):
    pos = np.clip(position, lower, upper).astype(int)
    if np.all((pos == 0)):
        pos[np.random.randint(0, len(pos))] = 1
    return pos


def fitness_function(solution):
    evaluator = Evaluator(train_X, test_X, train_Y, test_Y, solution, Config.CLASSIFIER, Config.DRAW_CONFUSION_MATRIX, Config.AVERAGE_METRIC)
    metrics = evaluator.get_metrics()
    if Config.PRINT_ALL:
        print(metrics)
    return list(metrics.values())      # Metrics return: [accuracy, precision, recall, f1]


data = get_dataset(Config.DATASET_NAME)
train_X, test_X, train_Y, test_Y = train_test_split(data.features, data.labels,
            stratify=data.labels, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE)
print(f"y_label: {test_Y}")

## 1. Define problem dictionary
n_features = data.features.shape[1]
n_labels = len(np.unique(data.labels))
LOWER_BOUND = [0, ] * n_features
UPPER_BOUND = [1.99, ] * n_features

problem = {
    "fit_func": fitness_function,
    "lb": LOWER_BOUND,
    "ub": UPPER_BOUND,
    "minmax": Config.MIN_MAX_PROBLEM,
    "obj_weights": Config.OBJ_WEIGHTS,
    "amend_position": amend_position,
}

## 2. Define algorithm and trial
model = DE.SHADE(problem, epoch=10, pop_size=20)
best_position, best_fitness = model.solve()
print(f"Best features: {best_position}, Best accuracy: {best_fitness}")


