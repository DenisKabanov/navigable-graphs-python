#!python3
# coding: utf-8

import numpy as np
from tqdm.auto import tqdm
from math import log2
from heapq import heappush, heappop
from sklearn.cluster import KMeans # для k-means кластеризации
from itertools import combinations # для перебора пар вершин
import random


def heuristic(candidates, curr, k, distance_func, data):
    candidates = sorted(candidates, key=lambda a: a[1])
    result_indx_set = {candidates[0][0]}
    result = [candidates[0]]
    added_data = [data[candidates[0][0]]]
    for c, curr_dist in candidates[1:]:
        c_data = data[c]       
        if curr_dist < min(map(lambda a: distance_func(c_data, a), added_data)):
            result.append( (c, curr_dist))
            result_indx_set.add(c)
            added_data.append(c_data)
    for c, curr_dist in candidates: # optional. uncomment to build neighborhood exactly with k elements.
        if len(result) < k and (c not in result_indx_set):
            result.append((c, curr_dist))
    return result

class HNSW_mod:
    # self._graphs[level][i] contains a {j: dist} dictionary,
    # где j это сосед i, а dist — расстояние до него
    # чем больше level — тем он выше в HNSW (в нём меньше точек) 

    def _distance(self, x, y): 
        return self.distance_func(x, [y])[0]

    def _vectorized_distance(self, x, ys): # функция для подсчёта расстояния от одной точки сразу до нескольких
        return [self.distance_func(x, y) for y in ys]

    # def __init__(self, distance_func, m=5, m0=None, k=64, ef=10, ef_construction=30, neighborhood_construction=heuristic, vectorized=False):
    def __init__(self, distance_func, m=5, m0=None, ef=10, ef_construction=30, neighborhood_construction=heuristic, vectorized=False):
        random.seed(108) # фиксируем параметр для генерации случайных чисел
        self.data = [] # список под данные
        self.distance_func = distance_func # функция подсчёта расстояния
        self.neighborhood_construction = neighborhood_construction # функция подсчёта окрестности
        print('Building HNSW...')

        # выбор, как считать расстояния между точками
        if vectorized:
            self.distance = self._distance
            self.vectorized_distance = distance_func
        else:
            self.distance = distance_func # функция подсчёта расстояния между парой точкек
            self.vectorized_distance = self._vectorized_distance # функция подсчёта расстояния между несколькими парами точками

        self._m = m # число стартовых точек
        self._m0 = 2 * m if m0 is None else m0
        # self.k = k # количество соседей у вершин (используется при модификации HNSW)
        self.k = self._m0 * 2 # количество соседей у вершин (используется при модификации HNSW)
        self._ef = ef
        self._ef_construction = ef_construction
        self._level_mult = 1 / log2(m)
        self._graphs = [] # список под подграфы
        self._enter_point = None # стартовая точка
        self.edges_extra = {} # словарь под дополнительные рёбра на нижнем уровне, по которым не будет идти перемещение, пока не приблизимся к оптимуму

    def add(self, elem, ef=None): # функция добавления новой вершины в HNSW
        if ef is None:
            ef = self._ef

        distance = self.distance # функция, используемая для подсчёта расстояния
        data = self.data # список координат точек, что уже есть в HNSW
        graphs = self._graphs # список подграфов
        point = self._enter_point # стартовая точка
        m = self._m # число стартовых точек

        level = int(-log2(random.random()) * self._level_mult) + 1 # уровень HNSW, на который будет вставлен элемент
        # print("level: %d" % level)

        # elem will be at data[idx]
        idx = len(data) # определяем id нового элемента
        data.append(elem) # добавляем его координаты в data


        if point is not None:  # HNSW не пуст <=> есть стартовая точка
            dist = distance(elem, data[point]) # считаем расстояние от добавляемого элемента до всех точек, что есть в графе
            
            # for all levels in which we dont have to insert elem, we search for the closest neighbor
            for layer in reversed(graphs[level:]): # идём, начиная с самого верхнего уровня (где меньше всего вершин)
                point, dist = self.beam_search(graph=layer, q=elem, k=1, eps=[point], ef=1)[0]
            
            # at these levels we have to insert elem; ep is a heap of entry points.
            layer0 = graphs[0]
            for layer in reversed(graphs[:level]):
                level_m = m if layer is not layer0 else self._m0
                # navigate the graph and update ep with the closest nodes we find
                # ep = self._search_graph(elem, ep, layer, ef)
                candidates = self.beam_search(graph=layer, q=elem, k=level_m*2, eps=[point], ef=self._ef_construction)
                point = candidates[0][0]
                
                # insert in g[idx] the best neighbors
                # layer[idx] = layer_idx = {}
                # self._select(layer_idx, ep, level_m, layer, heap=True)

                neighbors = self.neighborhood_construction(candidates=candidates, curr=idx, k=level_m, distance_func=self.distance_func, data=self.data)
                layer[idx] = neighbors
                # insert backlinks to the new node
                for j, dist in neighbors:
                    candidates_j = layer[j] + [(idx, dist)]
                    neighbors_j = self.neighborhood_construction(candidates=candidates_j, curr=j, k=level_m, distance_func=self.distance_func, data=self.data)
                    layer[j] = neighbors_j
                    
                
        for i in range(len(graphs), level):
            # for all new levels, we create an empty graph
            graphs.append({idx: []})
            self._enter_point = idx

    # can be used for search after jump
    def search(self, q, k=1, ef=10, level=0, return_observed=True, return_real_observed=False):
        graphs = self._graphs # список подграфов (графов на определённых уровнях)
        point = self._enter_point # входная точка
        #============================ initial search function ===================
        # for layer in reversed(graphs[level:]): # идём по уровням, начиная с самого верхнего (малозаполненного)
        #     point, dist = self.beam_search(layer, q=q, k=1, eps=[point], ef=1)[0] # находим ближайшую (?) точку на уровне к запросу q => она станет стартовой для следующего уровня
        # return self.beam_search(graph=graphs[level], q=q, k=k, eps=[point], ef=ef, return_observed=return_observed) # запускаем поиск на самом нижнем уровне со всеми вершинами
        #--------------- search function with calcs on layers -------------------
        observed_count = 0 # счётчик реального числа вызовов функции подсчёта дистанции
        for layer in reversed(graphs[level:]): # идём по уровням, начиная с самого верхнего (малозаполненного)
            observed = self.beam_search(layer, q=q, k=1, eps=[point], ef=1, return_observed=return_observed) # находим ближайшую (?) точку на уровне к запросу q => она станет стартовой для следующего уровня
            point, dist = observed[0] # ближайшая точка к запросу q
            observed_count += len(observed) # увеличиваем число реальных подсчётов дистанции на размер observed на текущем уровне HNSW
        res = self.beam_search(graph=graphs[level], q=q, k=k, eps=[point], ef=ef, return_observed=return_observed) # запускаем поиск на самом нижнем уровне со всеми вершинами
        observed_count += len(res) # добавляем количество подсчётов на самом нижнем уровне
        if return_real_observed:
            return res, observed_count # возвращаем ближайшие точки и число вызовов функции расстояния на уровнях HNSW
        else:
            return res # возвращаем ближайшие точки
        #========================================================================

    def beam_search(self, graph, q, k, eps, ef, ax=None, marker_size=20, return_observed=False): # работает аналогично K-графовской, но только на каждом уровне
        '''
        graph – the layer where the search is performed
        q - query
        k - number of closest neighbors to return
        eps – entry points [vertex_id, ..., vertex_id]
        ef – size of the beam
        return_observed – if True returns the full of elements for which the distance were calculated
        iters_extra — через сколько итераций хуже лучшей (точка дальше от оптимума) начинаить использовать дополнительные рёбра
        returns – a list of tuples [(vertex_id, distance), ... , ]
        '''
        # Priority queue: (negative distance, vertex_id)
        candidates = [] # список кандидатов для рассмотрения
        visited = set()  # set вершин, уже использованных для расширения списка кандидатов (candidates)
        observed = dict() # dict: vertex_id -> float – словарь вершин для которых было посчитанно растояние до запроса

        if ax:
            ax.scatter(x=q[0], y=q[1], s=marker_size, color='red', marker='^')
            ax.annotate('query', (q[0], q[1]))

        # Initialize the queue with the entry points
        for ep in eps:
            dist = self.distance_func(q, self.data[ep]) # считаем расстояние от стартовой точки до запроса
            heappush(candidates, (dist, ep)) # добавляем в кучу candidates данные о стартовой точке (расстояние, id точки) 
            observed[ep] = dist # запоминаем посчитанное расстояние

        while candidates: # пока имеются кандидаты
            # Get the closest vertex (furthest in the max-heap sense)
            dist, current_vertex = heappop(candidates) # берём самую верхнюю вершину кучи (ближайшую к запросу q)

            if ax:
                ax.scatter(x=self.data[current_vertex][0], y=self.data[current_vertex][1], s=marker_size, color='red')
                ax.annotate( len(visited), self.data[current_vertex] )

            # check stop conditions #####
            observed_sorted = sorted(observed.items(), key=lambda a: a[1]) # сортируем рассмотренные вершины по их расстоянию до запроса q
            # print(observed_sorted)
            ef_largets = observed_sorted[min(len(observed)-1, ef-1)] # самая удалённая точка от запроса q (но не дальше ef — максимального окна просмотра пройденных вершин)
            # print(ef_largets[0], '<->', -dist)
            if ef_largets[1] < dist: # если самая удалённая точка в окне ближе к запросу, чем текущая рассматриваемая
                break
            #############################

            visited.add(current_vertex) # добавляем текущую вершину в set посещённых

            # проверяем, нужно ли рассматривать дополнительные соседние вершины (не происходит, если не была проведена модификация графа)
            point_to_observe = graph[current_vertex].copy() # копируем соседей текущей вершины
            if (dist > observed_sorted[0][1]) and (len(graph) == len(self.edges_extra)): # если новое расстояние стало хуже лучшего найденного раньше и для уровня (самый нижний) у нас имеются дополнительные рёбра
                for closest_id, _ in observed_sorted[:2]: # идём по дополнительным вершинам найденных ближайших точек
                    point_to_observe += self.edges_extra[closest_id] # добавляем их экстра-соседей к соседям текущей вершины для рассмотрения

            # Check the neighbors of the current vertex
            for neighbor, _ in point_to_observe: # идём по соседям вершины на уровне
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

    def save_graph_plane(self, file_path):
        with open(file_path, "w") as f:
            f.write(f'{len(self.data)}\n')

            for x in self.data:
                s = ' '.join([a.astype('str') for a in x ])
                f.write(f'{s}\n')

            for graph in self._graphs:
                for src, neighborhood in graph.items():
                    for dst, dist in neighborhood: 
                        f.write(f'{src} {dst}\n')

    def modify_hnsw_1(self, max_closest=500):
        """
        Функция модификации HNSW для достижения лучших метрик (изменяет рёбра только на самом нижнем уровне, где представлены все вершины).\n
        Строит окрестности по правилу построения диаграммы Вороного. Также накидывает дополнительные рёбра, по которым идёт поиск только при близком нахождении к оптимуму.\n
        Parameters:
            * max_closest: сколько учитывать ближайших соседей при подсчёте, чем меньше значение — тем быстрее строится граф из-за меньшего перебора для проведения ребер (для малых размерностей (<10) достаточно достаточно брать max_closest=100, но в общем случае чем оно больше, тем "правильнее" будет граф, однако после определённого значения этот параметр лишь замедляет построение)\n
        Returns:
            * None: обновляет данные в графе
        """
        levels = [0] # рассматриваемые уровни для изменения
        for level in levels: # идём по изменяемым уровням
            nodes_at_level = self._graphs[level].keys() # вершины, встречаемые на уровне
            self._graphs[level] = {} # обнуляем данные о рёбрах на уровне
            # self.edges_extra[level] = {} # словарь под дополнительные рёбра для вершин на рассматриваемом уровне
            print(f"Modifying level {level}...")
            for x_id, x_coord in tqdm(enumerate(self.data)):
                self._graphs[level][x_id] = [] # список под смежные вершины для x
                other_points = sorted(enumerate(self._vectorized_distance(x_coord, self.data)), key=lambda a: a[1])[1:max_closest] # считаем дистанцию от x до остальных точек (1: — пропускаем нулевой элемент, так как это и есть сама вершина)
                other_points = [point for point in other_points if point[0] in nodes_at_level] # оставляем для рассмотрения только те точки, что уже есть на уровне

                self._graphs[level][x_id].append(other_points[0]) # добавляем ближайшего соседа, так как он всегда будет ближайшим (0 - сама точка)
                for y_id, y_dist in other_points[1:]: # идём по оставшимся точкам (пропущен 0, так как он уже добавлен)
                    if y_dist < min([self.distance_func(self.data[y_id], self.data[x_neighbor[0]]) for x_neighbor in self._graphs[level][x_id]]): # если рассматриваемая точки 'y' ближе к 'x', чем к какой-либо из соседей 'x'
                        self._graphs[level][x_id].append((y_id, y_dist))

                # добавляем дополнительные рёбра до k, по которым будем ходить только в том случае, если переход по уже добавленным не улучшает результат (расстояние до оптимума)
                edges_count = len(self._graphs[level][x_id]) # общее число рёбер у вершины
                connected_nodes = [point[0] for point in self._graphs[level][x_id]] # список id_шников уже соединённых с x вершин
                # self.edges_extra[level][x_id] = []
                self.edges_extra[x_id] = []
                for y_id, y_dist in other_points[2:]:
                    if edges_count >= self.k:
                        break
                    if y_id not in connected_nodes:
                        # self.edges_extra[level][x_id].append((y_id, y_dist))
                        self.edges_extra[x_id].append((y_id, y_dist))
                        edges_count += 1 # увеличиваем счётчик числа рёбер

    def modify_hnsw_2(self, cluster_prop=100):
        """
        Функция модификации HNSW для достижения лучших метрик (изменяет рёбра только на самом нижнем уровне, где представлены все вершины).\n
        Проводит дополнительные рёбра между медианами кластеров.\n
        Parameters:
            * cluster_prop: сколько относительно точек должно быть кластеров (100 => nodes/100 кластеров)\n
        Returns:
            * None: обновляет данные в графе
        """
        level = 0 # рассматриваемый уровень
        dim = len(self.data[0]) # размерность векторов, кодирующих вершины
        nodes_at_level = self._graphs[level].keys() # вершины, встречаемые на уровне 0
        n_clusters = max(int(len(nodes_at_level)/cluster_prop), 1)

        vertex_clusters = KMeans(n_clusters=n_clusters, # ожидаемое число кластеров
            max_iter=300, # максимальное число итераций K-means для одного запуска
            random_state=0, # случайное число для инициализации центроидов (centroid)
            ).fit_predict([self.data[node] for node in nodes_at_level]) # обучаем на данных и предсказываем их кластера

        # ищем медианы кластеров
        print("Modifying graph...")
        medians = [] # список под медианы кластеров
        for cluster_id in tqdm(range(n_clusters)): # идём по номерам кластеров
            points_in_cluster = np.where(np.array(vertex_clusters) == cluster_id)[0] # ищем индексы (id_шники) вершин в соответствующем кластере
            center = np.zeros(shape=(dim)) # центр кластера
            for point in points_in_cluster: # идём по точкам в кластере
                center += self.data[point] # добавляем их к центру
            center = center/len(points_in_cluster) # приходим к середине, поделив на число точек в кластере
            median = self.beam_search(self._graphs[level], q=center, k=1, eps=[self._enter_point], ef=1)[0][0] # ищем ближайшую к центру реальную вершину графа (только id)
            medians.append(median)

        for p_1, p_2 in combinations(medians, 2): # рассматриваем все комбинации стартовых вершин (центров кластеров)
            dist = self.distance_func(self.data[p_1], self.data[p_2]) # считаем расстояние между медианами
            self._graphs[level][p_1].append((p_2, dist))
            self._graphs[level][p_2].append((p_1, dist))

    def modify_hnsw_3(self, max_closest=500):
        """
        Функция модификации HNSW для достижения лучших метрик (изменяет рёбра только на самом нижнем уровне, где представлены все вершины).\n
        Строит окрестности по правилу построения диаграммы Вороного, итерационно — для получения длинных рёбер. Также накидывает дополнительные рёбра, по которым идёт поиск только при близком нахождении к оптимуму.\n
        Parameters:
            * cluster_prop: сколько относительно точек должно быть кластеров (100 => nodes/100 кластеров)\n
        Returns:
            * None: обновляет данные в графе
        """
        level = 0 # рассматриваемый уровень
        nodes_at_level = list(self._graphs[level].keys()) # вершины, встречаемые на уровне
        # self._graphs[level] = {node:list() for node in nodes_at_level} # обнуляем список под соседние вершины
        self._graphs[level] = {} # обнуляем список под соседние вершины

        # по очереди добавляем вершины в граф и соединяем их с уже имеющимися в нём (по правилу построения графа Вороного), получая длинные рёбра из-за поочерёдного добавления
        print("Modifying graph...")
        for x_id in tqdm(nodes_at_level): # идём по id вершинам на уровне графа
            other_nodes_at_level = list(self._graphs[level].keys()) # сколько в данный момент вершин на уровне
            self._graphs[level][x_id] = []
            if len(other_nodes_at_level) == 0: # если это первая вершина на уровне
                continue # переходим к добавлению остальных
            other_nodes_coord = np.array(self.data)[other_nodes_at_level] # координаты текущих точек в графе
            dists = self._vectorized_distance(self.data[x_id], other_nodes_coord) # считаем расстояние от рассматриваемой вершины до уже имеющихся в графе
            other_points = sorted(zip(other_nodes_at_level, dists), key=lambda a: a[1])[:max_closest] # сортируем по расстоянию от рассматриваемой вершины (от меньшего к большему)
            self._graphs[level][x_id].append(other_points[0]) # добавляем ближайшего соседа, так как он всегда будет ближайшим
            self._graphs[level][other_points[0][0]].append((x_id, other_points[0][1]))
            for y_id, y_dist in other_points[1:]: # идём по оставшимся точкам (пропущен 0, так как 'y' под индексом 0 всегда ближайшая к x)
                if y_dist < min([self.distance_func(self.data[y_id], self.data[x_neighbor[0]]) for x_neighbor in self._graphs[level][x_id]]): # если рассматриваемая точки 'y' ближе к 'x', чем к какой-либо из соседей 'x'
                    self._graphs[level][x_id].append((y_id, y_dist))
                    self._graphs[level][y_id].append((x_id, y_dist))

        # добавляем дополнительные рёбра, что были бы при построении графа Вороного 
        for x_id in tqdm(nodes_at_level): # идём по id вершинам на уровне графа
            other_points = sorted(enumerate(self._vectorized_distance(self.data[x_id], self.data)), key=lambda a: a[1])[1:max_closest]
            for y_id, y_dist in other_points: # идём по оставшимся точкам (пропущен 0, так как 'y' под индексом 0 всегда ближайшая к x)
                if y_dist < min([self.distance_func(self.data[y_id], self.data[x_neighbor[0]]) for x_neighbor in self._graphs[level][x_id]]): # если рассматриваемая точки 'y' ближе к 'x', чем к какой-либо из соседей 'x'
                    self._graphs[level][x_id].append((y_id, y_dist))
                    self._graphs[level][y_id].append((x_id, y_dist))
            self._graphs[level][x_id] = sorted(self._graphs[level][x_id], key=lambda a: a[1]) # сортируем получившиеся рёбра

        # добавляем дополнительные рёбра до k, по которым будем ходить только в том случае, если переход по уже добавленным не улучшает результат (расстояние до оптимума)
        print("Calc extra edges...")
        for x_id in tqdm(nodes_at_level): # идём по id вершинам на уровне графа
            other_points = sorted(enumerate(self._vectorized_distance(self.data[x_id], self.data)), key=lambda a: a[1])[1:max_closest] # считаем дистанцию от x до остальных точек (1: — пропускаем нулевой элемент, так как это и есть сама вершина)
            # other_points = [point for point in other_points if point[0] in nodes_at_level] # оставляем для рассмотрения только те точки, что уже есть на уровне
            edges_count = len(self._graphs[level][x_id]) # общее число рёбер у вершины
            connected_nodes = [point[0] for point in self._graphs[level][x_id]] # список id_шников уже соединённых с x вершин
            self.edges_extra[x_id] = []
            for y_id, y_dist in other_points:
                if edges_count >= self.k:
                    break
                if y_id not in connected_nodes:
                    self.edges_extra[x_id].append((y_id, y_dist))
                    edges_count += 1 # увеличиваем счётчик числа рёбер

    def modify_hnsw_4(self, k):
        """
        Функция модификации HNSW для достижения лучших метрик (изменяет рёбра только на самом нижнем уровне, где представлены все вершины).\n
        Случайным образом пересоединяет вершины с вероятностью, пропорциональной их удалённости.\n
        Parameters:
            * k: сколько случайных рёбер должно бытьу точки\n
        Returns:
            * None: обновляет данные в графе
        """
        level = 0 # рассматриваемый уровень
        nodes_at_level = list(self._graphs[level].keys()) # вершины, встречаемые на уровне
        self._graphs[level] = {node:list() for node in nodes_at_level} # обнуляем список под соседние вершины
        print("Modifying graph...")
        for x_id in tqdm(nodes_at_level): # идём по id вершинам на уровне графа
            other_points = sorted(enumerate(self._vectorized_distance(self.data[x_id], self.data)), key=lambda a: a[1])[1:]
            dists = [point[1] for point in other_points] # расстояния до точек
            neighbors = np.random.choice(range(len(other_points)), p=[dist/sum(dists) for dist in dists], size=k, replace=False) # выбираем, какие точки будут взяты (здесь считаются не НЕ ИХ id, а их положение в списке other_points)
            for i in neighbors:
                if other_points[i] not in self._graphs[level][x_id]:
                    self._graphs[level][x_id].append(other_points[i])
                    self._graphs[level][other_points[i][0]].append((x_id, other_points[i][1]))
        for x_id in tqdm(nodes_at_level):
            self._graphs[level][x_id] = sorted(self._graphs[level][x_id], key=lambda a: a[1])
        self.edges_extra = {} # не рассматриваем дополнительные рёбра
    
    def stat(self):
        for level in reversed(range(len(self._graphs))):
            print(f"На уровне {level} вершин {len(self._graphs[level])}.")
            avg_edges = 0
            nodes = len(self._graphs[level])
            for x in self._graphs[level].keys():
                avg_edges += len(self._graphs[level][x])
            avg_edges = avg_edges/nodes
            print(f"На уровне {level} среднее число рёбер: {avg_edges}.")

            if level == 0:
                for x in self.edges_extra.keys():
                    avg_edges += len(self.edges_extra[x])/nodes
                print(f"На уровне {level} среднее число рёбер (с учётом дополнительных рёбер): {avg_edges}.")

    def change_entry_point(self):
        print(f"Old enter: {self._enter_point}")
        print(self._graphs[-1])
        dim = len(self.data[0]) # размерность векторов, кодирующих вершины
        center = np.zeros(shape=(dim)) # центр кластера
        nodes_at_level = list(self._graphs[-1].keys())
        for id in nodes_at_level: # идём по точкам в кластере
            center += self.data[id] # добавляем их к центру
        center = center/len(nodes_at_level) # приходим к середине, поделив на число точек в кластере
        self._enter_point = self.beam_search(self._graphs[-1], q=center, k=1, eps=[self._enter_point], ef=50)[0][0] # ищем ближайшую к центру реальную вершину графа (только id)
        print(f"New enter: {self._enter_point}")


# import sys, os
# sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../")
# from utils import *

# # train_data, test_data = generate_synthetic_data(128, 2000, 100)
# # train_data, test_data = generate_synthetic_data(128, 1000, 100)
# train_data, test_data = generate_synthetic_data(3, 500, 100)
# # train_data, test_data = generate_synthetic_data(128, 100, 100)
# groundtruth_data = None
# recalls = {} # словарь под recall
# avg_calls = {} # словарь под среднее число подсчётов дистанции


# hnsw = HNSW_mod(distance_func=l2_distance, m=32, m0=64, ef=10, ef_construction=30, neighborhood_construction=heuristic)
# for x in tqdm(train_data): # добавляем точки в HNSW граф
#     hnsw.add(x)
# hnsw.stat()
# recalls["HNSW_0"], avg_calls["HNSW_0"] = calculate_recall_hnsw(l2_distance, hnsw, test_data, groundtruth_data, k=5, ef=20, m=1) # считаем метрики при построенном HNSW графе и запросах test_data, при условии что ответ — groundtruth_data

# hnsw.change_entry_point()
# recalls["HNSW_new_enter"], avg_calls["HNSW_new_enter"] = calculate_recall_hnsw(l2_distance, hnsw, test_data, groundtruth_data, k=5, ef=20, m=1) # считаем метрики при построенном HNSW графе и запросах test_data, при условии что ответ — groundtruth_data

# hnsw.modify_hnsw_1(max_closest=1000)
# hnsw.stat()
# recalls["HNSW_1"], avg_calls["HNSW_1"] = calculate_recall_hnsw(l2_distance, hnsw, test_data, groundtruth_data, k=5, ef=20, m=1) # считаем метрики при построенном HNSW графе и запросах test_data, при условии что ответ — groundtruth_data

# hnsw.modify_hnsw_2(cluster_prop=100)
# hnsw.stat()
# recalls["HNSW_2"], avg_calls["HNSW_2"] = calculate_recall_hnsw(l2_distance, hnsw, test_data, groundtruth_data, k=5, ef=20, m=1) # считаем метрики при построенном HNSW графе и запросах test_data, при условии что ответ — groundtruth_data

# hnsw.modify_hnsw_3()
# hnsw.stat()
# recalls["HNSW_3"], avg_calls["HNSW_3"] = calculate_recall_hnsw(l2_distance, hnsw, test_data, groundtruth_data, k=5, ef=20, m=1) # считаем метрики при построенном HNSW графе и запросах test_data, при условии что ответ — groundtruth_data

# hnsw.modify_hnsw_4(k=64)
# hnsw.stat()
# recalls["HNSW_4"], avg_calls["HNSW_4"] = calculate_recall_hnsw(l2_distance, hnsw, test_data, groundtruth_data, k=5, ef=20, m=1) # считаем метрики при построенном HNSW графе и запросах test_data, при условии что ответ — groundtruth_data


# for graph_type in recalls.keys():
#     print(f"Для {graph_type} средний recall: {recalls[graph_type]:.4f}, среднее число подсчётов расстояния: {avg_calls[graph_type]}.")
