
# coding: utf-8

# In[ ]:


import sys, math, random, heapq
import matplotlib.pyplot as plt
from itertools import chain
import random
import pandas as pd
import numpy as np
#from sklearn.cluster import KMeans

cities = pd.read_csv('test1.csv')
ver= cities[['x','y']].values.tolist()


x = cities['x'].values.tolist()
y = cities['y'].values.tolist()

verticex = zip(x,y)
city_list = list(verticex)


class Graph:
    def __init__(self,vertices):
        self.vertices = vertices
        self.n = len(vertices)
        
    def x(self, v):
        return self.vertices[v][0]

    def y(self, v):
        return self.vertices[v][1]

    # Lookup table for distances
    _d_lookup = {}

    def d(self, u, v):
        #this one to find the eucli distance between the vertices

        # Check if the distance was computed before
        if (u, v) in self._d_lookup:
            return self._d_lookup[(u, v)]

        # Otherwise compute it
        _distance = math.sqrt((u[0] - v[0])**2 + (u[1] - v[1])**2)

        # adding to the dict
        self._d_lookup[(u, v)], self._d_lookup[(v, u)] = _distance, _distance
        #print(self._d_lookup)
        return _distance
     

    def plot(self, tour=None):
        

        if tour is None:
            tour = Tour(self, [])

        _vertices = [self.vertices[0]]

        for i in tour.vertices:
            _vertices.append(self.vertices[i])

        _vertices.append(self.vertices[0])
        plt.figure(figsize=(20,20))
        plt.title("Cost = " + str(tour.cost()))      
        #img = plt.imread("map.jpg") #baigan, this doesn't work :(
        plt.scatter(*zip(*self.vertices), c="b", s=10, marker="s")    
        plt.plot(*zip(*_vertices), '-b')
        plt.show()
        #print(_d_lookup)
    
    
class Tour:

    def __init__(self, g, vertices = None):
        # constructor for this class
        self.g = g

        if vertices is None:
            self.vertices = list(range(1, g.n))
            random.shuffle(self.vertices)
        else:
            self.vertices = vertices

        self.__cost = None

    def cost(self):
# edge costs of the tour
        if self.__cost is None:
            self.__cost = 0
            for i, j in zip([0] + self.vertices, self.vertices + [0]):
                self.__cost += self.g.d(self.g.vertices[i], self.g.vertices[j])
        return self.__cost
    
class GeneticAlgorithm:

    def __init__(self, g, population_size, k=5, elite_mating_rate=2,
                 mutation_rate=1.015, mutation_swap_rate=1.2):
        
        self.g = g
        self.population = []
        
        for _ in range(population_size):
            self.population.append(Tour(g))

        self.population_size = population_size
        self.k = k
        self.elite_mating_rate = elite_mating_rate
        self.mutation_rate = mutation_rate
        self.mutation_swap_rate = mutation_swap_rate

    def crossover(self, p1, p2):
        size = len(p1.vertices)

        # Choose random start/end position for crossover
        a, b = [-1] * size, [-1] * size
        start, end = sorted([random.randrange(size) for _ in range(2)])

        # Replicate parent1's sequence for a, parent2's sequence for b
        for i in range(start, end + 1):
            a[i] = p1.vertices[i]
            b[i] = p2.vertices[i]

        # Fill the remaining position with the other parents' entries
        current_p2_position, current_p1_position = 0, 0

        for i in chain(range(start), range(end + 1, size)):

            while p2.vertices[current_p2_position] in a:
                current_p2_position += 1

            while p1.vertices[current_p1_position] in b:
                current_p1_position += 1

            a[i] = p2.vertices[current_p2_position]
            b[i] = p1.vertices[current_p1_position]

        # Return twins
        return Tour(self.g, a), Tour(self.g, b)

    def mutate(self, tour):
        """Randomly swaps pairs of cities in a given tour according to mutation rate"""

        # Decide whether to mutate
        if random.random() < self.mutation_rate:

            # For each vertex
            for i in range(len(tour.vertices)):

                # Randomly decide whether to swap
                if random.random() < self.mutation_swap_rate:

                    # Randomly choose other city position
                    j = random.randrange(len(tour.vertices))

                    # Swap
                    tour.vertices[i], tour.vertices[j] = tour.vertices[j], tour.vertices[i]

    def select_parent(self, k):
        """Implements k-tournament selection to choose parents"""
        tournament = random.sample(self.population, k)
        return max(tournament, key=lambda t: t.cost())
    
    def evolve(self):
        """Executes one iteration of the genetic algorithm to obtain a new generation"""

        new_population = []

        for _ in range(self.population_size):

            # K-tournament for parents
            p1, p2 = self.select_parent(self.k), self.select_parent(self.k)
            a, b = self.crossover(p1, p2)

            # Mate in an elite fashion according to the elitism_rate
            if random.random() < self.elite_mating_rate:
                if a.cost() < p1.cost() or a.cost() < p2.cost():
                    new_population.append(a)
                if b.cost() < p1.cost() or b.cost() < p2.cost():
                    new_population.append(b)

            else:
                self.mutate(a)
                self.mutate(b)
                new_population += [a, b]

        # Add new population to old
        self.population += new_population

        # Retain fittest
        self.population = heapq.nsmallest(self.population_size, self.population, key=lambda t: t.cost())
        

    def run(self, iterations=5000):
        for _ in range(iterations):
            self.evolve()                
        return min(self.population, key=lambda t: t.cost())
    
    def best(self):
        return max(self.population, key=lambda t: t.cost())
    
g = Graph(city_list)

ga = GeneticAlgorithm(g, 10)

ga.run()

best_tour = ga.best()
g.plot(best_tour)

#for distinct path(this is little funny)
#Since we din't know how start with finding the distinct path, one of our teammates came up with this.
#we just randomly shuffled the list and plotted it. :D


# In[ ]:



new_list = city_list
np.random.shuffle(new_list)

gr = Graph(new_list)
gnt = GeneticAlgorithm(gr, 10)

gnt.run()

best_tour = gnt.best()
gr.plot(best_tour)

