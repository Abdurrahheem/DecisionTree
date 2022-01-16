from __future__ import annotations
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd


def find_best_split(
    feature_vector: Union[np.ndarray, pd.DataFrame], 
    target_vector: Union[np.ndarray, pd.Series],
    criterion: str = "gini",
    feature_type: str = "real"
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    
    """
    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини или энтропии нужно выбирать порог с наименьшим значением.

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)
    :param criterion: либо `gini`, либо `entropy`
    :param feature_type: либо `real`, либо `categorical`
    
    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки
    :return q_values: вектор со значениями функционала ошибки для каждого из порогов в thresholds len(q_values) == len(thresholds)
    :return threshold_best: оптимальный порог
    :return q_value_best: значение функционала ошибки при оптимальном разбиении
    """
    
    size = feature_vector.shape[0]
    
    if feature_type == "real":
        s_f = np.sort(feature_vector)
        thresholds = np.unique(np.vstack([s_f[1:], s_f[:-1]]).mean(axis=0))
    else:
        thresholds = np.unique(feature_vector)
    
   
    ## использвание цыкла для перебора порогов(заменить на векторозованную форму)
    q_values = []
    for thr in thresholds:
        
        if feature_type == "real":
            idicies_left = np.where(feature_vector <= thr)[0]
            idicies_right = np.where(feature_vector > thr)[0]
        else:
            idicies_left = np.where(feature_vector == thr)[0]
            idicies_right = np.where(feature_vector != thr)[0]
        
        if criterion == 'gini':
            left_impurity = gini(target_vector[idicies_left])
            right_impurity = gini(target_vector[idicies_right])
        else:
            left_impurity = entropy(target_vector[idicies_left])
            right_impurity = entropy(target_vector[idicies_right])
         

        q = -((idicies_left.shape[0] * left_impurity) / size 
             + (idicies_right.shape[0] * right_impurity) / size)
        
        q_values.append(q)
        
    ## найти лучшие значения
    q_values = np.array(q_values)
    idx = np.argmax(q_values)  ## min or max?

    q_best = q_values[idx]
    threshold_best = thresholds[idx]
    
    return thresholds, q_values, threshold_best, q_best

def gini(targets):

    """Calculate the Gini Impurity.
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    
    size = targets.shape[0]
    unique, counts = np.unique(targets, return_counts=True)
    counts = dict(zip(unique, counts))

    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / size
        impurity -= prob_of_lbl**2
        
    return impurity    


def entropy(targets):
    
    """Calculated the Entropy Imputiry using:
    https://www.section.io/engineering-education/entropy-information-gain-machine-learning/
    """
    
    size = targets.shape[0]
    unique, counts = np.unique(targets, return_counts=True)
    counts = dict(zip(unique, counts))
    
    impurity = 0
    for lbl in counts:
        prob_of_lbl = counts[lbl] / size
        impurity += prob_of_lbl * np.log(prob_of_lbl)
    
    return -impurity
