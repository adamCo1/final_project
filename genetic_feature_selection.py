import random
import numpy as np
import pandas as pd
import sklearn.metrics
from operator import itemgetter
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

class Mutual_Information_Estimator():

    def __init__(self, dataset, target, features_names):
        self.dataset = dataset
        self.features_vectors = dataset
        self.class_vector = target
        self.features_names = features_names

    def calculate_score(self, features_names):
        total_score = 0
        for feature in features_names:
            total_score = total_score + sklearn.metrics.mutual_info_score(
                self.class_vector, self.features_vectors[feature], contingency=None)
        return total_score







class GeneticSelector():
    # TODO find good enough parameters :  n_gen size, n_best n_rand, n_children mutation_rate
    def __init__(self, estimator, num_of_generations, num_of_chromosomes,  num_best_chromosomes, num_rand_chromosomes,
                 num_crossover_children, features_names, operator_probability):
        self.estimator = estimator
        self.features_names = features_names
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
         sorted_scores = None
         X, y = self.dataset
         scores = []
         for chromosome_index, chromosome in enumerate(population):
            columns_mask = []
            for index, value in enumerate(chromosome):
                if value == True:
                    columns_mask.append(self.features_names[index])
            data_to_fit = X[columns_mask]
            score = self.estimator.calculate_score(columns_mask)
            scores.append([chromosome_index,score])
         scores.sort(key=itemgetter(1), reverse=True)

         return scores, population

    def select_best_chromosomes(self, scores_sorted, population_sorted):
        population_next = []
        for chromosome_num in range(self.num_best_chromosomes):
            count = 0
            population_next.append(population_sorted[scores_sorted[chromosome_num][0]])
        for i in range(self.num_rand_chromosomes):
            population_next.append(random.choice(population_sorted))
        return population_next

    def crossover(self, population):
        population_next = population
        while(len(population_next) < self.num_of_chromosomes):
            for i in range(int(len(population_next) / 2), len(population)):
                for j in range(self.num_crossover_children ):
                    chromosome1, chromosome2 = population[i], population[len(population) - 1 - j]
                    child = chromosome1
                    mask = np.random.rand(len(child)) > 0.7
                    child[mask] = chromosome2[mask]
                    population_next.append(child)
        return population_next

    def mutate(self, population):

        population_next = []
        for i in range(len(population)):
            chromosome = population[i]
            if(self.__should_apply_operator__()):
                mask = np.random.rand(len(chromosome)) < 0.3
                chromosome[mask] = False
            population_next.append(chromosome)
        return population_next

    def generate_population(self, population):
        # Selection, crossover and mutation
        sorted_scores, population = self.fitness(population)
        population = self.select_best_chromosomes(sorted_scores, population)
        #if(self.__should_apply_operator__()):
        #population = self.__duplicate_chromosomes__(population)
        #if(self.__should_apply_operator__()):
        population = self.crossover(population)
        #if(self.__should_apply_operator__()):
        #population = self.mutate(population)
        # History
        #self.chromosomes_best.append(population_sorted[0])
        #self.scores_best.append(scores_sorted[0])
        #self.scores_avg.append(np.mean(scores_sorted))
        print(sorted_scores[0][1])
        self.best = population[0]
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

def main():
    data_df = pd.read_csv('wdbc.csv', sep=',')
    data_vector, target_vector, features_names = data_df[data_df.columns[1:-1]], data_df[data_df.columns[-1]], list(data_df.columns.values)
    for i in range(20):
        mu = Mutual_Information_Estimator(data_vector, target_vector, features_names)
        selector = GeneticSelector(estimator = mu,
                                   num_of_generations = 10,
                                   num_of_chromosomes = 200,
                                   num_best_chromosomes = 70,
                                   num_rand_chromosomes = 10,
                                   num_crossover_children = 5,
                                   features_names = features_names[1:-1],
                                   operator_probability = 0.1)
        selector.fit(data_vector, target_vector)
        best_features = selector.best
        columns_mask = []
        for index, value in enumerate(best_features):
            if value == True:
                columns_mask.append(data_df.columns.values[1:-1][index])
        best_score = mu.calculate_score(columns_mask)
        print(i, best_features, best_score)


main()
