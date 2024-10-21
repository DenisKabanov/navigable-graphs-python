#!python3

import numpy as np
import random
from tqdm.auto import tqdm
random.seed(108)


def read_fvecs(filename):
    with open(filename, 'rb') as f:
        while True:
            vec_size = np.fromfile(f, dtype=np.int32, count=1)
            if not vec_size:
                break
            vec = np.fromfile(f, dtype=np.float32, count=vec_size[0])
            yield vec


def read_ivecs(filename):
    with open(filename, 'rb') as f:
        while True:
            vec_size = np.fromfile(f, dtype=np.int32, count=1)
            if not vec_size:
                break
            vec = np.fromfile(f, dtype=np.int32, count=vec_size[0])
            yield vec


def load_sift_dataset():
    train_file = 'datasets/siftsmall/siftsmall_base.fvecs'
    test_file = 'datasets/siftsmall/siftsmall_query.fvecs'
    groundtruth_file = 'datasets/siftsmall/siftsmall_groundtruth.ivecs'

    train_data = np.array(list(read_fvecs(train_file)))
    test_data = np.array(list(read_fvecs(test_file)))
    groundtruth_data = np.array(list(read_ivecs(groundtruth_file)))

    return train_data, test_data, groundtruth_data


def generate_synthetic_data(dim, n, nq):
    train_data = np.random.random((n, dim)).astype(np.float32)
    test_data = np.random.random((nq, dim)).astype(np.float32)
    return train_data, test_data


def l2_distance(a, b):
    return np.linalg.norm(a - b) # считаем евклидово расстояние

def brute_force_knn_search(distance_func, k, q, data):
    """
    Функция для точного нахождения ближайших k точек к запросу q.\n
    Parameters
    ----------
    * distance_func : функция подсчёта расстояния между точками
    * k : число искомых ближайших соседей на этапе тестирования
    * q : координаты точки запроса
    * data : данные о координатах всех точек графа\n
    Returns
    ----------
    * list : (idx, dist) for k-closest elements to {x} in {data}
    """
    return sorted(enumerate(map(lambda x: distance_func(q, x) ,data)), key=lambda a: a[1])[:k]

def calculate_recall(kg, test, groundtruth, k, ef, m):
    """
    Функция подсчёта метрик для K-графов.\n
    Parameters
    -------
    * kg : HNSW граф
    * test : список координат запросов
    * groundtruth : список id точек правильных ответов (k ближайших точкек для всех запросов из test)
    * k : сколько точек брать ближайших из рассмотренных для ответа
    * ef : параметр выхода из функции поиска (size of the beam for beam search)
    * m : число стартовых точек\n
    Returns
    -------
    * tuple : (среднее значение метрики recall, среднее количество вызовов подсчёта расстояния)
    """
    if groundtruth is None:
        print("Ground truth not found. Calculating ground truth...")
        groundtruth = [ [idx for idx, dist in kg.brute_force_knn_search(k, query)] for query in tqdm(test)]

    print("Calculating recall...")
    recalls = []
    total_calc = 0
    for query, true_neighbors in tqdm(zip(test, groundtruth), total=len(test)):
        true_neighbors = true_neighbors[:k]  # Use only the top k ground truth neighbors
        if hasattr(kg, "entry_points"): # проверяем, есть ли у графа приоритетные стартовые точки
            entry_points = random.sample(kg.entry_points, m)
        else: # если нет — берём случайные m точек
            entry_points = random.sample(range(len(kg.data)), m)
        entry_points = random.sample(range(len(kg.data)), m)
        observed = [neighbor for neighbor, dist in kg.beam_search(query, k, entry_points, ef, return_observed=True)]
        total_calc = total_calc + len(observed)
        results = observed[:k]
        intersection = len(set(true_neighbors).intersection(set(results)))
        # print(f'true_neighbors: {true_neighbors}, results: {results}. Intersection: {intersection}')
        recall = intersection / k
        recalls.append(recall)

    return np.mean(recalls), total_calc/len(test)

def calculate_recall_hnsw(distance_func, kg, test, groundtruth, k, ef, m):
    """
    Функция подсчёта метрик для HNSW.\n
    Parameters
    -------
    * distance_func : функция подсчёта расстояния
    * kg : HNSW граф
    * test : список координат запросов
    * groundtruth : список id точек правильных ответов (k ближайших точкек для всех запросов из test)
    * k : сколько точек брать ближайших из рассмотренных для ответа
    * ef : параметр выхода из функции поиска (size of the beam for beam search)
    * m : число стартовых точек\n
    Returns
    -------
    * tuple : (среднее значение метрики recall, среднее количество вызовов подсчёта расстояния)
    """
    if groundtruth is None:
        print("Ground truth not found. Calculating ground truth...")
        groundtruth = [ [idx for idx, dist in brute_force_knn_search(distance_func, k, query, kg.data)] for query in tqdm(test)] # для каждого запроса из test ищем id-шники (idx) ближайших k точек, что есть в графе (kg.data)

    print("Calculating recall...")
    recalls = []
    total_calc = 0
    for query, true_neighbors in tqdm(zip(test, groundtruth), total=len(test)):
        true_neighbors = true_neighbors[:k]  # Use only the top k ground truth neighbors
        # entry_points = random.sample(range(len(kg.data)), m)
        observed = [neighbor for neighbor, dist in kg.search(q=query, k=k, ef=ef, return_observed=True)]
        total_calc = total_calc + len(observed)
        results = observed[:k]
        intersection = len(set(true_neighbors).intersection(set(results)))
        # print(f'true_neighbors: {true_neighbors}, results: {results}. Intersection: {intersection}')
        recall = intersection / k
        recalls.append(recall)

    return np.mean(recalls), total_calc/len(test) # возвращаем метрики

def calculate_recall_hnsw_real(distance_func, kg, test, groundtruth, k, ef, m):
    """
    Функция подсчёта метрик для HNSW с учётом реального количества вызовов функции расстояния.\n
    Parameters
    -------
    * distance_func : функция подсчёта расстояния
    * kg : HNSW граф
    * test : список координат запросов
    * groundtruth : список id точек правильных ответов (k ближайших точкек для всех запросов из test)
    * k : сколько точек брать ближайших из рассмотренных для ответа
    * ef : параметр выхода из функции поиска (size of the beam for beam search)
    * m : число стартовых точек\n
    Returns
    -------
    * tuple : (среднее значение метрики recall, среднее количество вызовов подсчёта расстояния)
    """
    if groundtruth is None:
        print("Ground truth not found. Calculating ground truth...")
        groundtruth = [[idx for idx, dist in brute_force_knn_search(distance_func, k, query, kg.data)] for query in tqdm(test)] # для каждого запроса из test ищем id-шники (idx) ближайших k точек, что есть в графе (kg.data)

    print("Calculating recall...")
    recalls = []
    total_calc = 0
    for query, true_neighbors in tqdm(zip(test, groundtruth), total=len(test)):
        true_neighbors = true_neighbors[:k]  # Use only the top k ground truth neighbors
        # entry_points = random.sample(range(len(kg.data)), m)
        observed, calcs = kg.search(q=query, k=k, ef=ef, return_observed=True, return_real_observed=True)
        observed = [neighbor for neighbor, dist in observed]
        total_calc = total_calc + calcs
        results = observed[:k]
        intersection = len(set(true_neighbors).intersection(set(results)))
        # print(f'true_neighbors: {true_neighbors}, results: {results}. Intersection: {intersection}')
        recall = intersection / k
        recalls.append(recall)

    return np.mean(recalls), total_calc/len(test) # возвращаем метрики