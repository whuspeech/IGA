import numpy as np
import utils
import models
import random

seed = 4044
random.seed(seed)
np.random.seed(seed)

# 超参数
population_size = 200
alpha = 1.0
beta = 0.0
delta = 0.05
crossover_prob = 0.8
pr = 0.5
mutation_prob = 0.2
memory_size = 0
radius = 3
crossover_mode = 'MSCX_Radius'

iterations = 6000
exp_num = 7

data_path = 'data/att48.tsp'
save_path = f'experiments/GA-{exp_num}-{crossover_mode}-{population_size}-{iterations}.npy'

# initialization
cities, city_num = utils.get_cities(data_path)
distance_matrix = utils.get_distance_matrix(cities, city_num)

Population = models.Population(chro_num=population_size,
                               gene_num=city_num,
                               distance_matrix=distance_matrix,
                               alpha=alpha,
                               beta=beta,
                               delta=delta,
                               crossover_prob=crossover_prob,
                               mutation_prob=mutation_prob,
                               memory_size=memory_size,
                               radius=radius,
                               crossover_mode=crossover_mode)

Population.init_population()
distances = []  # 存储每代的最优结果，用于可视化

# 迭代
for i in range(iterations):
    # 选择
    Population.selection()

    # 交叉
    Population.crossover()

    # 变异
    Population.mutation()

    # 记忆
    Population.memorization()

    # 更新
    Population.update()

    # 记录每代的最短距离
    best_seq = Population.best_chromosome.gene_seq
    best_distance = utils.compute_distance(best_seq, city_num, distance_matrix)
    distances.append(best_distance)

    print(Population.generation)

utils.save_result(distances, save_path)
utils.visualize_result(distances)
