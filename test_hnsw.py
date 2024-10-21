#!/usr/bin/env python
# coding: utf-8

import argparse
from utils import *
from graphs.hnsw import HNSW, heuristic
import random
random.seed(108)


def main():
    parser = argparse.ArgumentParser(description='Test recall of beam search method with KGraph.')
    parser.add_argument('--dataset', choices=['synthetic', 'sift'], default='synthetic', help="Choose the dataset to use: 'synthetic' or 'sift'.")
    parser.add_argument('--K', type=int, default=5, help='The size of the neighbourhood')
    parser.add_argument('--M', type=int, default=50, help='Avg number of neighbors')
    parser.add_argument('--M0', type=int, default=50, help='Avg number of neighbors')
    parser.add_argument('--dim', type=int, default=2, help='Dimensionality of synthetic data (ignored for SIFT).')
    parser.add_argument('--n', type=int, default=200, help='Number of training points for synthetic data (ignored for SIFT).')
    parser.add_argument('--nq', type=int, default=50, help='Number of query points for synthetic data (ignored for SIFT).')
    parser.add_argument('--k', type=int, default=5, help='Number of nearest neighbors to search in the test stage')
    parser.add_argument('--ef', type=int, default=10, help='Size of the beam for beam search.')
    parser.add_argument('--m', type=int, default=3, help='Number of random entry points.')
    args = parser.parse_args()

    # Load dataset
    if args.dataset == 'sift': # загружаем датасет sift
        print("Loading SIFT dataset...")
        train_data, test_data, groundtruth_data = load_sift_dataset()
    else: # генерирем свой датасет
        print(f"Generating synthetic dataset with {args.dim}-dimensional space...")
        train_data, test_data = generate_synthetic_data(args.dim, args.n, args.nq)
        groundtruth_data = None

    # Create HNSW
    hnsw = HNSW(distance_func=l2_distance, m=args.M, m0=args.M0, ef=10, ef_construction=30, neighborhood_construction=heuristic)
    for x in tqdm(train_data): # добавляем точки в HNSW граф
        hnsw.add(x)

    # Calculate recall
    recall, avg_cal = calculate_recall_hnsw(l2_distance, hnsw, test_data, groundtruth_data, k=args.k, ef=args.ef, m=args.m) # считаем метрики при построенном HNSW графе и запросах test_data, при условии что ответ — groundtruth_data
    print(f"Average recall: {recall}, avg calc: {avg_cal}")

if __name__ == "__main__":
    main()
