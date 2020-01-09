import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

SEED = 2018
random.seed(SEED)
np.random.seed(SEED)


# ==============================================================================
# Data
# ==============================================================================
# dataset = load_boston()
# X, y = dataset.data, dataset.target
# features = dataset.feature_names

# TODO : simple GUI
# TODO clean data

# ==============================================================================
# CV MSE before feature selection
# ==============================================================================
# TODO create Mutual Information estimator class
# estimator = mutual_information_estimator()
# TODO  create score formula
# score = -1.0 * cross_val_score(est, X, y, cv=5, scoring="neg_mean_squared_error")


# print("CV MSE before feature selection: {:.2f}".format(np.mean(score)))


# ==============================================================================
# Class performing feature selection with genetic algorithm
# ==============================================================================
class GeneticSelector():
    # TODO find good enough parameters :  n_gen size, n_best n_rand, n_children mutation_rate
    def __init__(self, estimator, num_of_generations, num_of_chromosomes,  num_best_chromosomes, num_rand_chromosomes,
                 num_crossover_children, operator_probability):
        self.estimator = estimator(dataset)
        self.num_of_generations = num_of_generations
        self.num_of_chromosomes = num_of_chromosomes
        self.num_best_chromosomes = num_best_chromosomes
        self.num_rand_chromosomes = num_rand_chromosomes
        self.num_crossover_children = num_crossover_children
        self.operator_probability = operator_probability
        self.__checkPopulationSize__();

    def __checkPopulationSize__(self):
        best_to_random = (self.num_best_chromosomes + self.num_rand_chromosomes) / 2
        if(int(best_to_random * self.num_crossover_children) != self.num_of_chromosomes) :
            raise ValueError('population not stable')

    def initialize_population(self):
        population = []
        for index in range(self.num_of_chromosomes):
            chromosome = np.ones(self.n_features, dtype=np.bool)
            mask = np.random.rand(len(chromosome)) < 0.3
            chromosome[mask] = False
            population.append(chromosome)
        return population

    # TODO fitness calculation function
    def fitness(self, population):
         X, y = self.dataset
         scores = []
         for index, chromosome in enumerate(population):
            data_to_fit = X[:, chromosome]
            if data_to_fit.shape[1] > 0 :
                score = -1.0 * np.mean(cross_val_score(self.estimator, X[:,chromosome], y,
                                        cv=5,
                                        scoring="neg_mean_squared_error"))
                score = estimator.calculate_score(chromsome)
                scores.append(score)
            else:
                scores.append(0)

         scores, population = np.array(scores), np.array(population)
         inds = np.argsort(scores)
         return list(scores[inds]), list(population[inds,:])

    def select_best_chromosomes(self, population_sorted):
        population_next = []
        for index in range(self.num_best_chromosomes):
            population_next.append(population_sorted[index])
        for i in range(self.num_rand_chromosomes):
            population_next.append(random.choice(population_sorted))
        random.shuffle(population_next)
        return population_next

    def crossover(self, population):
        # TODO : no reason for O(N^2). implement in other way
        population_next = []
        for i in range(int(len(population) / 2)):
            for j in range(self.num_crossover_children ):
                chromosome1, chromosome2 = population[i], population[len(population) - 1 - i]
                child = chromosome1
                mask = np.random.rand(len(child)) > 0.8
                child[mask] = chromosome2[mask]
                population_next.append(child)
        return population_next

    def mutate(self, population):

        population_next = []
        for i in range(len(population)):
            chromosome = population[i]
            mask = np.random.rand(len(chromosome)) < 0.1
            chromosome[mask] = False
            population_next.append(chromosome)
        return population_next

    def generate_population(self, population):
        # Selection, crossover and mutation
        scores_sorted, population_sorted = self.fitness(population)
        population = self.select_best_chromosomes(population_sorted)
        #if(self.__should_apply_operator__()):
        #population = self.__duplicate_chromosomes__(population)
        #if(self.__should_apply_operator__()):
        population = self.crossover(population)
        #if(self.__should_apply_operator__()):
        population = self.mutate(population)
        # History
        self.chromosomes_best.append(population_sorted[0])
        self.scores_best.append(scores_sorted[0])
        self.scores_avg.append(np.mean(scores_sorted))

        return population

    def __duplicate_chromosomes__(self, population):
        next_population = population
        for times in range(self.num_rand_chromosomes):
            next_population.append(random.choice(population))
        return next_population

    def fit(self, data_vector, target_vector):

        self.chromosomes_best = []
        self.scores_best, self.scores_avg = [], []

        self.dataset = data_vector, target_vector
        self.n_features = data_vector.shape[1]

        population = self.initialize_population()
        for i in range(self.num_of_generations):
            population = self.generate_population(population)

        return self

    def __should_apply_operator__(self):
        return random.random() < self.operator_probability

# def main():
#     dataset = load_boston()
#     data_vector, target_vector = dataset.data, dataset.target
#     features = dataset.feature_names
#     for i in range(20):
#         selector = GeneticSelector(estimator = LinearRegression(),
#                                    num_of_generations = 70,
#                                    num_of_chromosomes = 200,
#                                    num_best_chromosomes = 40,
#                                    num_rand_chromosomes = 40,
#                                    num_crossover_children = 5,
#                                    operator_probability = 0.1)
#         selector.fit(data_vector, target_vector)
#         best_features = selector.chromosomes_best[0]
#         print(i,best_features)
#
#
# main()