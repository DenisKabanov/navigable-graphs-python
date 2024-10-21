#!/usr/bin/env python
# coding: utf-8

import argparse
from utils import *
from graphs.navigable_graphs import KmGraph
from graphs.hnsw import HNSW, heuristic
from graphs.hnsw_mod import HNSW_mod
from graphs.voronoi_graph import GraphVoronoi
import random
random.seed(108)

def main():
    parser = argparse.ArgumentParser(description="Тестирование графов для поиска ближайшего соседа к запросу по метрике 'recall' и среднему число вызовов подсчёта расстояния.")
    parser.add_argument('--dataset', choices=['synthetic', 'sift'], default='synthetic', help="Какой датасет использовать: 'synthetic' или 'sift'.")
    parser.add_argument('--dim', type=int, default=2, help='Размерность векторов генерируемых (synthetic) точек (ignored for SIFT).')
    parser.add_argument('--n', type=int, default=200, help='Количество генерируемых (synthetic) точек (вершин) для графа (ignored for SIFT).')
    parser.add_argument('--nq', type=int, default=50, help="Количество генерируемых (synthetic) точек 'запроса' (ignored for SIFT).")
    parser.add_argument('--K', type=int, default=5, help='Количество соседей у вершин при построении графа.')
    parser.add_argument('--k', type=int, default=5, help='Количество ближаших соседей к запросу, что нужно найти.')
    parser.add_argument('--M0', type=int, default=50, help='Среднее число рёбер на нижнем уровне HNSW (где все вершины), влияет на количество уровней в графе.')
    parser.add_argument('--M', type=int, default=50, help='Число случайных рёбер для KmGraph и среднее число соседей для HNSW, влияет на количество уровней в графе.')
    parser.add_argument('--m', type=int, default=3, help='Число входных (начальных, от которых пойдёт поиск) точек.')
    parser.add_argument('--ef', type=int, default=10, help='Size of the beam for beam search.')
    args = parser.parse_args()

    # загружаем датасет
    if args.dataset == 'sift':
        print("Loading SIFT dataset...")
        train_data, test_data, groundtruth_data = load_sift_dataset()
    else:
        print(f"Generating synthetic dataset with {args.dim}-dimensional space...")
        train_data, test_data = generate_synthetic_data(args.dim, args.n, args.nq)
        groundtruth_data = None


    recalls = {} # словарь под recall
    avg_calls = {} # словарь под среднее число подсчётов дистанции

    # тестирование Km Graph
    kg = KmGraph(k=args.K, dim=args.dim, dist_func=l2_distance, data=train_data, M=args.M)
    recalls["KmGraph"], avg_calls["KmGraph"] = calculate_recall(kg, test_data, groundtruth_data, k=args.k, ef=args.ef, m=args.m)
    print(f"Средний recall: {recalls['KmGraph']:.4f}, среднее число подсчётов расстояния: {avg_calls['KmGraph']}.")

    # тестирование Graph Voronoi
    vg = GraphVoronoi(k=args.K, dist_func=l2_distance, data=train_data)
    recalls["GraphVoronoi"], avg_calls["GraphVoronoi"] = calculate_recall(vg, test_data, groundtruth_data, k=args.k, ef=args.ef, m=args.m)
    print(f"Средний recall: {recalls['GraphVoronoi']:.4f}, среднее число подсчётов расстояния: {avg_calls['GraphVoronoi']}.")

    # тестирование HNSW
    hnsw = HNSW(distance_func=l2_distance, m=args.M, m0=args.M0, ef=10, ef_construction=30, neighborhood_construction=heuristic)
    for x in tqdm(train_data): # добавляем точки в HNSW граф
        hnsw.add(x)
    recalls["HNSW"], avg_calls["HNSW"] = calculate_recall_hnsw(l2_distance, hnsw, test_data, groundtruth_data, k=args.k, ef=args.ef, m=args.m) # считаем метрики при построенном HNSW графе и запросах test_data, при условии что ответ — groundtruth_data
    print(f"Средний recall: {recalls['HNSW']:.4f}, среднее число подсчётов расстояния: {avg_calls['HNSW']}.")

    hnsw_ = HNSW_mod(distance_func=l2_distance, m=args.M, m0=args.M0, ef=10, ef_construction=30, neighborhood_construction=heuristic)
    for x in tqdm(train_data): # добавляем точки в HNSW граф
        hnsw_.add(x)
    recalls["HNSW real"], avg_calls["HNSW real"] = calculate_recall_hnsw_real(l2_distance, hnsw_, test_data, groundtruth_data, k=args.k, ef=args.ef, m=args.m) # считаем метрики при построенном HNSW графе и запросах test_data, при условии что ответ — groundtruth_data
    print(f"Средний recall: {recalls['HNSW real']:.4f}, среднее число подсчётов расстояния: {avg_calls['HNSW real']}.")

    # hnsw_.modify_hnsw_1(max_closest=1000) # проводим первую модификацию — изменяем принцип построения рёбер на нижнем уровне 
    # # hnsw_.modify_hnsw_2(cluster_prop=100) # проводим вторую модификацию — ищем кластеры на нижнем уровне и соединяем их медианы
    # # hnsw_.modify_hnsw_3(max_closest=1000) # проводим вторую модификацию — изменяем принцип построения рёбер на нижнем уровне 
    # # hnsw_.modify_hnsw_4(max_closest=1000) # проводим вторую модификацию — изменяем принцип построения рёбер на нижнем уровне 
    # recalls["HNSW_mod"], avg_calls["HNSW_mod"] = calculate_recall_hnsw_real(l2_distance, hnsw_, test_data, groundtruth_data, k=args.k, ef=args.ef, m=args.m) # считаем метрики при построенном HNSW графе и запросах test_data, при условии что ответ — groundtruth_data
    

if __name__ == "__main__":
    main()
