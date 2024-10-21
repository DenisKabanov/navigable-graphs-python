#!python3

import numpy as np
from tqdm.auto import tqdm
from heapq import heappush, heappop
from sklearn.cluster import KMeans # для k-means кластеризации
from itertools import combinations # для перебора пар вершин
import random

class GraphVoronoi(object):
    def __init__(self, k, dist_func, data, max_closest=1000, cluster_prop=100):
        """
        Конструктор графа Вороного с проведение дополнительных рёбер.\n
        Parameters:
            * k: количество соседей (дополнительных) у вершин
            * dist_func: функция расстояния
            * data: координаты вершин (список списков)
            * max_closest: сколько учитывать ближайших соседей при подсчёте, чем меньше значение — тем быстрее строится граф из-за меньшего перебора для проведения ребер (для малых размерностей (<10) достаточно достаточно брать max_closest=100, но в общем случае чем оно больше, тем "правильнее" будет граф, однако после определённого значения этот параметр лишь замедляет построение)
            * cluster_prop: сколько относительно точек должно быть кластеров (100 => nodes/100 кластеров)\n
        Returns:
            * None: обновляет данные в графе
        """
        random.seed(108) # фиксируем параметр для генерации случайных чисел
        self.distance_func = dist_func # функция подсчёта расстояния
        self.data = data # точки графа (список списков)
        self.k = k # количество соседей у вершин
        self.dim = len(data[0]) # размерность векторов, кодирующих вершины
        print('Building Voronoi graph...')

        self.edges = [] # рёбра графа, по которым идёт обычное перемещение
        self.edges_extra = [] # рёбра графа, которые рассматриваем в случае достижения "оптимальной" точки
        for x_id, x_coord in tqdm(enumerate(self.data)):
            other_points = sorted(enumerate(self._vectorized_distance(x_coord, self.data)), key=lambda a: a[1])[1:max_closest] # пропускаем нулевой элемент, так как это и есть сама вершина
            self.edges.append([other_points[0]]) # добавляем ближайшего соседа, так как он всегда будет ближайшим (0 - сама точка)
            for y_id, y_dist in other_points[1:]: # идём по оставшимся точкам (пропущен 0, так как он уже добавлен, ибо ближайший к x)
                if y_dist < min([self.distance_func(data[y_id], data[x_neighbor[0]]) for x_neighbor in self.edges[x_id]]): # если рассматриваемая точки 'y' ближе к 'x', чем к какой-либо из соседей 'x'
                    self.edges[x_id].append((y_id, y_dist))

            # добавляем дополнительные рёбра до k, по которым будем ходить только в том случае, если переход по уже добавленным не улучшает результат (расстояние до оптимума)
            edges_count = len(self.edges[x_id]) # общее число рёбер у вершины
            self.edges_extra.append([])
            for y_id, y_dist in other_points[1:]:
                if edges_count >= k:
                    break
                if y_id not in self.edges[x_id]:
                    self.edges_extra[x_id].append((y_id, y_dist))
                    edges_count += 1 # увеличиваем счётчик числа рёбер

        self.k_means_clustering(n_clusters=max(int(len(self.data)/cluster_prop), 1)) # проводим кластеризацию вершин (на k кластеров, k>=1), добавляя длинные рёбра и стартовые точки

        avg_edges = 0
        nodes = len(self.edges)
        for i in range(nodes):
            avg_edges += len(self.edges[i])
            # avg_edges += len(self.edges[i]) + len(self.edges_extra[i])
        print(f"Среднее число рёбер у вершин: {avg_edges/nodes}.")


    def beam_search(self, q, k, eps, ef, ax=None, marker_size=20, return_observed=False):
        """
        q - query
        k - number of closest neighbors to return
        eps – стартовые точки ~ entry points [vertex_id, ..., vertex_id]
        ef – максимальное окно для просмотра пройденных вершин (используется для выхода из поиска)
        observed – if True returns the full of elements for which the distance were calculated
        returns – a list of tuples [(vertex_id, distance), ... , ]
        """
        # Priority queue: (negative distance, vertex_id)
        candidates = [] # список кандидатов для рассмотрения
        visited = set()  # set вершин, уже использованных для расширения списка кандидатов (candidates)
        observed = dict() # dict: vertex_id -> float – словарь вершин для которых было посчитанно растояние до запроса

        if ax:
            ax.scatter(x=q[0], y=q[1], s=marker_size, color='red', marker='^')
            ax.annotate('query', (q[0], q[1]))

        # инициализируем очередь с помощью стартовых точек
        for ep in eps:
            dist = self.distance_func(q, self.data[ep]) # считаем расстояние от стартовой точки до запроса
            heappush(candidates, (dist, ep)) # добавляем в кучу candidates данные о стартовой точке (расстояние, id точки) 
            observed[ep] = dist # запоминаем посчитанное расстояние

        while candidates: # пока имеются кандидаты
            # Get the closest vertex (furthest in the max-heap sense)
            dist, current_vertex = heappop(candidates) # берём самую верхнюю вершину кучи (ближайшую к запросу q)

            if ax:
                ax.scatter(x=self.data[current_vertex][0], y=self.data[current_vertex][1], s=marker_size, color='red')
                ax.annotate(len(visited), self.data[current_vertex])

            # check stop conditions #####
            observed_sorted = sorted(observed.items(), key=lambda a: a[1]) # сортируем рассмотренные вершины по их расстоянию до запроса q
            # print(observed_sorted)
            ef_largets = observed_sorted[min(len(observed)-1, ef-1)] # самая удалённая точка от запроса q (но не дальше ef — максимального окна просмотра пройденных вершин)
            # print(ef_largets[0], '<->', -dist)
            if ef_largets[1] < dist: # если самая удалённая точка в окне ближе к запросу, чем текущая рассматриваемая
                break
            #############################

            visited.add(current_vertex) # добавляем текущую вершину в set посещённых

            # проверяем, нужно ли рассматривать дополнительные соседние вершины
            point_to_observe = self.edges[current_vertex].copy() # копируем соседей текущей вершины
            if dist > observed_sorted[0][1]: # если новое расстояние стало хуже лучшего найденного раньше
                for closest_id, _ in observed_sorted[:10]: # идём по дополнительным вершинам найденных ближайших точек
                    point_to_observe += self.edges_extra[closest_id] # добавляем их экстра-соседей к соседям текущей вершины для рассмотрения

            # рассматриваем потенциально близкие вершины к запросу
            for neighbor, _ in point_to_observe:
                if neighbor not in observed: # если сосед ещё не рассмотрен
                    dist = self.distance_func(q, self.data[neighbor]) # считаем расстояние от соседа до запроса
                    heappush(candidates, (dist, neighbor)) # добавляем данные о соседе в кучу
                    observed[neighbor] = dist # запоминаем расстояние
                    if ax:
                        ax.scatter(x=self.data[neighbor][0], y=self.data[neighbor][1], s=marker_size, color='yellow')
                        # ax.annotate(len(visited), (self.data[neighbor][0], self.data[neighbor][1]))
                        ax.annotate(len(visited), self.data[neighbor])

        # Sort the results by distance and return top-k
        observed_sorted = sorted(observed.items(), key=lambda a: a[1])
        if return_observed:
            return observed_sorted
        return observed_sorted[:k]
    
    def _vectorized_distance(self, x, ys):
        return [self.distance_func(x, y) for y in ys]

    def brute_force_knn_search(self, k, x):
        '''
        Return the list of (idx, dist) for k-closest elements to {x} in {data}
        '''
        return sorted(enumerate(self._vectorized_distance(x, self.data)), key=lambda a: a[1])[:k]
    
    def plot_graph(self, ax, color, linewidth=0.5):
        ax.scatter(self.data[:, 0], self.data[:, 1], c=color)
        for i in range(len(self.data)):
            for edge_end in self.edges[i]:
                ax.plot( [self.data[i][0], self.data[edge_end][0]], [self.data[i][1], self.data[edge_end][1]], c=color, linewidth=linewidth )

    def k_means_clustering(self, n_clusters):
        """
        Метод для нахождения разбиения точек на заданное количество кластеров (далее центры (медианы) кластеров будут использованы в качестве начальных точек).\n
        Parameters:
            * n_clusters: количество кластеров, на которые нужно разбить данные\n
        Returns:
            * None: обновляет данные в графе
        """
        vertex_clusters = KMeans(n_clusters=n_clusters, # ожидаемое число кластеров
               max_iter=300, # максимальное число итераций K-means для одного запуска
               random_state=0, # случайное число для инициализации центроидов (centroid)
              ).fit_predict(self.data) # обучаем на данных и предсказываем их кластера

        # ищем медианы кластеров
        self.entry_points = [] # стартовые точки для поиска (id_шники медиан кластеров)
        for cluster_id in range(n_clusters): # идём по номерам кластеров
            points_in_cluster = np.where(np.array(vertex_clusters) == cluster_id)[0] # ищем индексы (id_шники) вершин в соответствующем кластере
            center = np.zeros(shape=(self.dim)) # центр кластера
            for point in points_in_cluster: # идём по точкам в кластере
                center += self.data[point] # добавляем их к центру
            center = center/len(points_in_cluster) # приходим к середине, поделив на число точек в кластере
            median = self.brute_force_knn_search(k=1, x=center)[0][0] # ищем ближайшую к центру реальную вершину графа (только id)
            self.entry_points.append(median) # добавляем id медианы кластера как стартовую точку

        for p_1, p_2 in combinations(self.entry_points, 2): # рассматриваем все комбинации стартовых вершин (центров кластеров)
            dist = self.distance_func(self.data[p_1], self.data[p_2]) # считаем расстояние между медианами
            self.edges[p_1].append((p_2, dist))
            self.edges[p_2].append((p_1, dist))

# import sys, os
# sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../")
# from utils import *
# train_data, test_data = generate_synthetic_data(3, 500, 100)

# vg = GraphVoronoi(k=20, dist_func=l2_distance, data=train_data, m=5)
# vg.k_means_clustering()