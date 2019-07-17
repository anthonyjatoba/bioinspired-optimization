import struct
import random
import textwrap
import numpy as np
from random import randint, uniform
from math import isnan
from functions import sphere, rastringin, ackley


def float_to_bin(num):
    return format(struct.unpack('!I', struct.pack('!f', num))[0], '032b')


def bin_to_float(binary):
    return struct.unpack('!f', struct.pack('!I', int(binary, 2)))[0]


class GA:
    def __init__(self, fun, bounds, generations=100, pop_size=50, cx_prob=.85, cx_strategy='one-point', mt_prob=0, sel_strategy='elitist'):
        self.generations = generations      # Number of generations
        self.pop_size = pop_size            # Population size
        self.cx_prob = cx_prob              # Probability of two parents procreate
        self.fun = fun                      # Function to optimize
        self.bounds = bounds                # Problem boundaries
        self.mt_prob = mt_prob              # Probability that a bit is flipped over

        if cx_strategy == 'one-point':
            self.cx_strategy = self.one_point_cx
        elif cx_strategy == 'two-point':
            self.cx_strategy = self.two_point_cx
        
        if sel_strategy == 'roulette':
            self.sel_strategy = self.roulette_selection
        elif sel_strategy == 'random':
            self.sel_strategy = self.random_selection
        else:
            self.sel_strategy = self.elitist_selection
        # TODO: roulette, random

    def gen_individual(self):
        """ Generates an individual binary string chromossome respecting the problem boundaries """
        b_values = ''
        for b in self.bounds:
            # random float inside the boundary
            value = random.uniform(b[0], b[1])
            # converts to a binary value (a string)
            b_values = b_values + float_to_bin(value)
        return b_values

    def sort_pop(self, population):
        """ Sorts a population in ascending order of fitness """
        return sorted(population, key=lambda i: i['fitness'])

    def evaluate(self, population):
        """ Calculates the fitness for each individual in a population"""
        for ind in population:
            # Breaks the string into strings of 32 bits (a float binary)
            binary_values = textwrap.wrap(ind['chromosome'], 32)
            
            for i in range(len(binary_values)):
                if binary_values[i][1] == '1' and binary_values[i][2] == '1' and binary_values[i][3] == '1' and \
                binary_values[i][4] == '1' and binary_values[i][5] == '1' and binary_values[i][6] == '1' and \
                binary_values[i][7] == '1' and binary_values[i][8] == '1':
                    v1 = list(binary_values[i])
                    v1[8] = '0'
                    binary_values[i] = ''.join(v1)
                    
            # Converts the binaries to floats
            float_values = [bin_to_float(v) for v in binary_values]
            ind['fitness'] = self.fun(float_values)
            if isnan(ind['fitness']):
                print(ind)
        return population

    def mutate(self, population):
        """ Mutates individuals by flipping its bits given a mutation rate """
        for individual in population:
            chromosome_mutated = ''
            for c in individual['chromosome']:
                if random.random() < self.mt_prob:
                    chromosome_mutated += '1' if c == '0' else '0'
                else:
                    chromosome_mutated += c
            individual['chromosome'] = chromosome_mutated
        return population

    def one_point_cx(self, ind1, ind2):
        """ One point crossover """
        ch_len = len(ind1['chromosome'])

        # Crossover point
        point = random.randrange(1,ch_len)

        ch1_chromosome = ind1['chromosome'][0:point] + \
            ind2['chromosome'][point:ch_len]
        ch2_chromosome = ind2['chromosome'][0:point] + \
            ind1['chromosome'][point:ch_len]

        child1 = {'chromosome': ch1_chromosome, 'fitness': -np.inf,
                  'inv_fitness': -np.inf, 'norm_fitness': -np.inf, 
                  'cumulative_fitness': -np.inf, 'chosen': False}
        child2 = {'chromosome': ch2_chromosome, 'fitness': -np.inf,
                  'inv_fitness': -np.inf, 'norm_fitness': -np.inf,
                  'cumulative_fitness': -np.inf, 'chosen': False}

        return child1, child2
    
    def two_point_cx(self,ind1,ind2):
        """ Two Point Crossover """
        ch_len = len(ind1['chromosome'])
        #crossover points
        point1 = random.randrange(1,ch_len)
        point2 = random.randrange(point1,ch_len)
        
        ch1_chromosome = ind1['chromosome'][0:point1] + \
            ind2['chromosome'][point1:point2] + \
            ind1['chromosome'][point2:ch_len]
        ch2_chromosome = ind2['chromosome'][0:point1] + \
            ind1['chromosome'][point1:point2] + \
            ind2['chromosome'][point2:ch_len]
        
        child1 = {'chromosome': ch1_chromosome, 'fitness': -np.inf,
                  'inv_fitness': -np.inf, 'norm_fitness': -np.inf, 
                  'cumulative_fitness': -np.inf, 'chosen': False}
        child2 = {'chromosome': ch2_chromosome, 'fitness': -np.inf,
                  'inv_fitness': -np.inf, 'norm_fitness': -np.inf,
                  'cumulative_fitness': -np.inf, 'chosen': False}
        return child1, child2

    def parents_selection(self, population):
        """ Randomly selects two parents from a population"""
        parent1 = population[random.randrange(self.pop_size)]
        parent2 = population[random.randrange(self.pop_size)]
        return parent1, parent2

    def elitist_selection(self, population):
        """ Selects the (pop_size) best individuals from a population"""
        population = self.sort_pop(population)
        return population[0:self.pop_size]
    
    def roulette_selection(self, population):
        """ Selects (pop_size) best individuals from a population using roulette selection"""
        
        total_fitness = 0
        cumulative_fitness = 0
        new_population = []
        for individual in population:
            individual['inv_fitness'] = 1/(individual['fitness'] + 1)
            total_fitness += individual['inv_fitness']
            
        for individual in population:               
            individual['norm_fitness'] = individual['inv_fitness']/total_fitness
            cumulative_fitness += individual['norm_fitness']
            individual['cumulative_fitness'] = cumulative_fitness
        
        while len(new_population) != self.pop_size:
            rand = uniform(0, 1)
            for individual in population:
                if rand <= individual['cumulative_fitness']:
                    if not individual['chosen']:
                        new_population.append(individual)
                    break
        
        return new_population
    
    def random_selection(self, population):
        """ Randomly selects the (pop_size) individuals"""
        new_population = []
        
        while len(new_population) != self.pop_size:
            rand = randint(0, len(population) - 1)
            if not population[rand]['chosen']:
                new_population.append(population[rand])
                population[rand]['chosen'] = True
       
        return new_population

    def crossover(self, population):
        """ Performs crossover into a population until a offspring with (pop_size) individuals is completed """
        offspring = []
        while len(offspring) < self.pop_size:
            parent1, parent2 = self.parents_selection(population)
            if random.random() < self.cx_prob:
                child1, child2 = self.cx_strategy(parent1, parent2)
                offspring.extend([child1, child2])
        self.evaluate(offspring)
        return offspring

    def run(self):
        # Initialize the population
        population = [{'chromosome': self.gen_individual(), 'fitness': -np.inf,
                       'inv_fitness': -np.inf, 'norm_fitness': -np.inf, 
                       'cumulative_fitness': -np.inf, 'chosen': False}
                      for x in range(self.pop_size)]
        
        # Initializing array for best fitness in each generation
        gen_best_fitnesses = []
        
        # Calculate the fitness for each individual in the initial population
        self.evaluate(population)
        
        # Sorts initial population
        population = self.sort_pop(population)
            
        for g in range(self.generations):
            # Generating the offspring
            offspring = self.crossover(population)

            # Selecting the new population from the old + offpring
            population = self.sel_strategy(population + offspring)
            
            # Reseting statuses for next generation
            for i in range(self.pop_size):
                population[i]['chosen'] = False
            
            if self.mt_prob > 0:
                # Perform evolutionary operations (mutation, etc.)
                population = self.mutate(population)
            
            # Calculate the fitness for each individual
            self.evaluate(population)
            
            # Sorts population
            population = self.sort_pop(population)
            
            # Saves generation best fitness
            gen_best_fitnesses.append(population[0]['fitness'])
            
        return population[0]['fitness'], gen_best_fitnesses