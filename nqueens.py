# -*- coding: utf-8 -*-
import random, math
from collections import Counter


class Solver_8_queens:
    def __init__(self, pop_size=1000, cross_prob=0.8, mut_prob=0.4):
        self.population_size = pop_size

        self.selector = RouletteSelection()
        self.crossover = Crossover(cross_prob)
        self.mutator = Mutator(mut_prob)
        self.visualizer = СhessboardVisualizer()

    def solve(self, min_fitness=0.9, max_epochs=50000):
        if max_epochs is None: max_epochs = float('inf')
        if min_fitness is None: min_fitness = float('inf')

        epoch_number = 0

        population = [Individual() for i in range(self.population_size + 100)]

        while True:
            epoch_number += 1
            population = self.selector.select_individuals(population, self.population_size)
            descendants = self.crossover.cross_population(population)
            self.mutator.mutation_population(descendants)
            population += descendants

            best_fitness = max(individual.fitness for individual in population)

            if best_fitness >= min_fitness or epoch_number == max_epochs:
                result = [individual for individual in population
                          if individual.fitness == best_fitness]
                visualization = self.visualizer.get_field(result[0].chromosome)
                break

        return best_fitness, epoch_number, visualization


class Individual:
    def __init__(self, chromosome=None):
        if chromosome is not None:
            self.chromosome = chromosome
        else:
            self.chromosome = self._generate_chromosome()

        self.fitness = self._calculate_fitness(self.chromosome)

    def update_fitness(self):
        self.fitness = self._calculate_fitness(self.chromosome)

    def _generate_chromosome(self, gene_number=8):
        return [random.randint(0, gene_number - 1) for i in range(0, gene_number)]

    def _calculate_fitness(self, chromosome):
        horizontal_conflicts = self._horizontal_conflicts(chromosome)
        diagonal_conflicts = self._diagonal_conflicts(chromosome)
        conflicts = horizontal_conflicts + diagonal_conflicts
        return (1 + conflicts) ** -1

    def _horizontal_conflicts(self, chromosome):
        return sum([i - 1 for i in Counter(chromosome).values() if i != 1])


    def _diagonal_conflicts(self, chromosome):
        conflicts = 0

        for i in range(len(chromosome)):
            for j in range(i + 1, len(chromosome)):
                if math.fabs(j - i) == math.fabs(chromosome[j] - chromosome[i]):
                    conflicts += 1

        return conflicts


class RouletteSelection:
    def select_individuals(self, population, count):
        roulette = self._get_roulette(population)
        return [self._select_individual(roulette) for i in range(0, count)]

    def _select_individual(self, roulete):
        selected = random.random()
        for individual in roulete:
            if selected <= individual[1]:
                return individual[0]

    def _get_roulette(self, population):
        probabilities = self._get_probabilities(population)
        roulette = [probabilities[0]]

        for i in range(1, len(probabilities)):
            roulette.append((probabilities[i][0],
                             roulette[i - 1][1] + probabilities[i][1]))

        return roulette

    def _get_probabilities(self, population):
        fitness_sum = sum([individual.fitness for individual in population])
        return [(individual, individual.fitness / fitness_sum)
                for individual in population]


class Crossover:
    def __init__(self, cross_prob):
        self.cross_prob = cross_prob

    def cross_population(self, population):
        children = []

        for individual in population:
            probability = random.random()
            if probability <= self.cross_prob:
                second_parent = self._get_parent(population)
                children += self.cross_parent(individual, second_parent)

        return children

    def cross_parent(self, first_parent, second_parent, crossing_point=3):
        first_chromosome = first_parent.chromosome[:crossing_point] + \
                           second_parent.chromosome[crossing_point:]
        second_chromosome = second_parent.chromosome[:crossing_point] + \
                            first_parent.chromosome[crossing_point:]
        return Individual(first_chromosome), Individual(second_chromosome)

    def _get_parent(self, population):
        parent = population[random.randint(0, len(population) - 1)]
        return parent


class Mutator:
    def __init__(self, mutation_probability):
        self.mutation_probability = mutation_probability

    def mutation_population(self, population):
        for i in range(len(population)):
            if random.random() <= self.mutation_probability:
                population[i] = self.mutation_individual(population[i])
                population[i].update_fitness()

    def mutation_individual(self, individual):
        gene_index = random.randint(0, len(individual.chromosome) - 1)
        new_gen = (individual.chromosome[gene_index] + 1) % \
                   len(individual.chromosome)
        individual.chromosome[gene_index] = new_gen
        return individual


class СhessboardVisualizer:
    def get_field(self, chromosome, filler='+', queen='Q'):
        field = ['{0}{1}{2}'.format(filler*gene,
                                    queen,
                                    filler * (len(chromosome) - gene - 1))
                 for gene in chromosome]

        return '\n'.join(field)
