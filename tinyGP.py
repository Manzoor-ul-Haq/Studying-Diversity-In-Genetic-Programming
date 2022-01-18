# tiny genetic programming by © moshe sipper, www.moshesipper.com
from random import random, randint, seed
from statistics import mean
from copy import deepcopy
import matplotlib.pyplot as plt
import sys
import numpy as np
from os import path
import seaborn as sns
import importlib
from plotnine import *
import pandas as pd

POP_SIZE        = 60   # population size
MIN_DEPTH       = 2    # minimal initial random tree depth
MAX_DEPTH       = 5    # maximal initial random tree depth
GENERATIONS     = 250  # maximal number of generations to run evolution
TOURNAMENT_SIZE = 5    # size of tournament for tournament selection
XO_RATE         = 0.8  # crossover rate
PROB_MUTATION   = 0.2  # per-node mutation probability

def add(x, y): return x + y
def sub(x, y): return x - y
def mul(x, y): return x * y
FUNCTIONS = [add, sub, mul]
TERMINALS = ['x', -2, -1, 0, 1, 2]

def target_func(x): # evolution's target
    return x*x*x*x + x*x*x + x*x + 2

def generate_dataset(): # generate 101 data points from target_func
    dataset = []
    arrayX = []
    arrayY = []
    for x in range(-100,101,2): 
        x /= 100
        arrayX.append(x)
        t = target_func(x)
        arrayY.append(t)
        dataset.append([x, t])
    return arrayX, arrayY, dataset

# man, zoor, ul = generate_dataset()
# plt.plot(man, zoor)
# plt.show()

#sys.exit()

class GPTree:
    postOrderedList = []
    def __init__(self, data = None, left = None, right = None):
        self.data  = data
        self.left  = left
        self.right = right
        
    def node_label(self): # string label
        if (self.data in FUNCTIONS):
            return self.data.__name__
        else: 
            return str(self.data)
    
    def print_tree(self, prefix = ""): # textual printout
        print("%s%s" % (prefix, self.node_label()))        
        if self.left:  self.left.print_tree (prefix + "   ")
        if self.right: self.right.print_tree(prefix + "   ")

    def compute_tree(self, x):
        if (self.data in FUNCTIONS):
            return self.data(self.left.compute_tree(x), self.right.compute_tree(x))
        elif self.data == 'x': return x
        else: return self.data
            
    def random_tree(self, grow, max_depth, depth = 0): # create random tree using either grow or full method
        if depth < MIN_DEPTH or (depth < max_depth and not grow): 
            self.data = FUNCTIONS[randint(0, len(FUNCTIONS)-1)]
        elif depth >= max_depth:   
            self.data = TERMINALS[randint(0, len(TERMINALS)-1)]
        else: # intermediate depth, grow
            if random () > 0.5: 
                self.data = TERMINALS[randint(0, len(TERMINALS)-1)]
            else:
                self.data = FUNCTIONS[randint(0, len(FUNCTIONS)-1)]
        if self.data in FUNCTIONS:
            self.left = GPTree()          
            self.left.random_tree(grow, max_depth, depth = depth + 1)         
            self.right = GPTree()
            self.right.random_tree(grow, max_depth, depth = depth + 1)

    def mutation(self):
        if random() < PROB_MUTATION: # mutate at this node
            self.random_tree(grow = True, max_depth = 2)
        elif self.left: self.left.mutation()
        elif self.right: self.right.mutation() 

    def size(self): # tree size in nodes
        if self.data in TERMINALS: return 1
        l = self.left.size()  if self.left  else 0
        r = self.right.size() if self.right else 0
        return 1 + l + r

    def build_subtree(self): # count is list in order to pass "by reference"
        t = GPTree()
        t.data = self.data
        if self.left:  t.left  = self.left.build_subtree()
        if self.right: t.right = self.right.build_subtree()
        return t
                        
    def scan_tree(self, count, second): # note: count is list, so it's passed "by reference"
        count[0] -= 1            
        if count[0] <= 1: 
            if not second: # return subtree rooted here
                return self.build_subtree()
            else: # glue subtree here
                self.data  = second.data
                self.left  = second.left
                self.right = second.right
        else:  
            ret = None              
            if self.left  and count[0] > 1: ret = self.left.scan_tree(count, second)  
            if self.right and count[0] > 1: ret = self.right.scan_tree(count, second)  
            return ret

    def crossover(self, other): # xo 2 trees at random nodes
        if random() < XO_RATE:
            second = other.scan_tree([randint(1, other.size())], None) # 2nd random subtree
            self.scan_tree([randint(1, self.size())], second) # 2nd subtree "glued" inside 1st tree

    def children(self, node):
        if node:
            self.children(node.left)
            self.children(node.right)
            self.postOrderedList.append(node)

    def childrenPostOrderedList(self, node):
        self.children(node)
        childrenList = self.postOrderedList
        self.postOrderedList = []
        return childrenList

    def tuplesSubtree(self, tree):
        T = self.childrenPostOrderedList(tree)
        
        subTrees = []

        for i in range(len(T)):
            if T[i].left and T[i].right:
                subTrees.append((T[i].left.data, T[i].right.data, T[i].data))
            elif T[i].left:
                subTrees.append((T[i].left.data, T[i].data))
            elif T[i].right:
                subTrees.append((T[i].right.data, T[i].data))

        return subTrees

    def union(self, subtreesTree1, subtreesTree2):
        union = []

        temp1 = subtreesTree1
        temp2 = subtreesTree2

        i, j = 0, 0

        while(i < len(temp1)):
            while(j < len(temp2)):
                if temp1[i] == temp2[j]:
                    t1 = temp1.pop(temp1.index(temp1[i]))
                    t2 = temp2.pop(temp2.index(temp2[j]))
                    union.append(t1)
                    i = -1
                    j = 0
                    break
                j += 1
            i += 1

        for i in range(len(temp1)):
            union.append(temp1[i])

        for i in range(len(temp2)):
            union.append(temp2[i])
        
        return union

    def intersection(self, subtreesTree1, subtreesTree2):
        intersection = []

        for i in range(len(subtreesTree1)):
            if subtreesTree1[i] in subtreesTree2:
                intersection.append(subtreesTree1[i])

        return intersection

    def jaccardIndex(self, tree1, tree2):
        subtreesTree1 = self.tuplesSubtree(tree1)
        subtreesTree2 = self.tuplesSubtree(tree2)

        intersection = self.intersection(subtreesTree1, subtreesTree2)
        union = self.union(subtreesTree1, subtreesTree2)

        if (not subtreesTree1 and not subtreesTree2) and (not intersection and not union):
            return 1
        
        return len(intersection) / len(union)

    def jaccard_similarityMatrix(self, list1):
        matrix = np.zeros((len(list1), len(list1)))
        
        list2 = list1

        for i in range(len(list1)):
            for j in range(len(list2)):
                matrix[i][j] = self.jaccardIndex(list1[i], list2[j])

        return matrix

    def treedist(self, i, j):
        i_children = self.childrenPostOrderedList(i)
        j_children = self.childrenPostOrderedList(j)

        distances = np.zeros((len(i_children) + 1, len(j_children) + 1))

        for t1 in range(len(i_children) + 1):
            distances[t1][0] = t1

        for t2 in range(len(j_children) + 1):
            distances[0][t2] = t2

        a = 0
        b = 0
        c = 0

        for t1 in range(1, len(i_children) + 1):
            for t2 in range(1, len(j_children) + 1):
                if i_children[t1 - 1].data == j_children[t2 - 1].data:
                    distances[t1][t2] = distances[t1 - 1][t2 - 1]
                else:
                    a = distances[t1][t2 - 1]
                    b = distances[t1 - 1][t2]
                    c = distances[t1 - 1][t2 - 1]

                    if a <= b and a <= c:
                        distances[t1][t2] = a + 1
                    elif b <= a and b <= c:
                        distances[t1][t2] = b + 1
                    else:
                        distances[t1][t2] = c + 1

        # self.printDistances(distances, len(i_children), len(j_children))
        return distances[len(i_children), len(j_children)]

    def ted_similarityMatrix(self, list1):
        matrix = np.zeros((len(list1), len(list1)))
        
        list2 = list1

        for i in range(len(list1)):
            for j in range(len(list2)):
                matrix[i][j] = self.treedist(list1[i], list2[j])

        for i in range(len(list1)):
            for j in range(len(list2)):
                matrix[i][j] /= -matrix.max()
                matrix[i][j] += 1
        
        # print(matrix[20][19])
        # sys.exit()
        return matrix

# end class GPTree
                   
def init_population(): # ramped half-and-half
    pop = []
    for md in range(3, MAX_DEPTH + 1):
        for i in range(int(POP_SIZE/6)):
            t = GPTree()
            t.random_tree(grow = True, max_depth = md) # grow
            pop.append(t) 
        for i in range(int(POP_SIZE/6)):
            t = GPTree()
            t.random_tree(grow = False, max_depth = md) # full
            pop.append(t) 
    return pop

def fitness(individual, dataset): # inverse mean absolute error over dataset normalized to [0,1]
    return 1 / (1 + mean([abs(individual.compute_tree(ds[0]) - ds[1]) for ds in dataset]))
                
def selection(population, fitnesses): # select one individual using tournament selection
    tournament = [randint(0, len(population)-1) for i in range(TOURNAMENT_SIZE)] # select tournament contenders
    tournament_fitnesses = [fitnesses[tournament[i]] for i in range(TOURNAMENT_SIZE)]
    return deepcopy(population[tournament[tournament_fitnesses.index(max(tournament_fitnesses))]]) 
            
def main():
    # init stuff
    seed() # init internal state of random number generator
    man, zoor, dataset = generate_dataset()
    population= init_population()
    t = GPTree()
    ted = t.ted_similarityMatrix(population)
    jaccard = t.jaccard_similarityMatrix(population)

    # columns = []
    # for i in range(60):
    #     columns.append("T" + str(i+1))
    # print(columns)

    outpath = "C:/Users/admin/Documents/Namal/Fall 2021/CSE-491 Final Year Project-1/Studying-Diversity-In-Genetic-Programming/GraphsTED"
    
    #graphs
    ax = sns.heatmap(ted, linewidth=0.5, vmin=0, vmax=1)
    ax.set_title("Generation 0")
    plt.savefig(path.join(outpath,"Generation_0.png"))
    
    outpath = "C:/Users/admin/Documents/Namal/Fall 2021/CSE-491 Final Year Project-1/Studying-Diversity-In-Genetic-Programming/GraphsJaccard"

    ax = sns.heatmap(jaccard, linewidth=0.5, vmin=0, vmax=1, cbar=False)
    ax.set_title("Generation 0")
    plt.savefig(path.join(outpath,"Generation_0.png"))

    # sys.exit()

    best_of_run = None
    best_of_run_f = 0
    best_of_run_gen = 0
    fitnesses = [fitness(population[i], dataset) for i in range(POP_SIZE)]
    counter = 0
    # go evolution!
    for gen in range(GENERATIONS):  
        nextgen_population=[]
        for i in range(POP_SIZE):
            parent1 = selection(population, fitnesses)
            parent2 = selection(population, fitnesses)
            parent1.crossover(parent2)
            parent1.mutation()
            nextgen_population.append(parent1)
        population = nextgen_population
        counter += 1
        ted = t.ted_similarityMatrix(population)
        jaccard = t.jaccard_similarityMatrix(population)
    
        outpath = "C:/Users/admin/Documents/Namal/Fall 2021/CSE-491 Final Year Project-1/Studying-Diversity-In-Genetic-Programming/GraphsTED"

        #Graphs
        ax = sns.heatmap(ted, linewidth=0.5, vmin=0, vmax=1, cbar=False)
        ax.set_title("Generation " + str(counter))
        plt.savefig(path.join(outpath,"Generation_" + str(counter) + ".png"))

        outpath = "C:/Users/admin/Documents/Namal/Fall 2021/CSE-491 Final Year Project-1/Studying-Diversity-In-Genetic-Programming/GraphsJaccard"

        ax = sns.heatmap(jaccard, linewidth=0.5, vmin=0, vmax=1, cbar=False)
        ax.set_title("Generation " + str(counter))
        plt.savefig(path.join(outpath,"Generation_" + str(counter) + ".png"))
        
        fitnesses = [fitness(population[i], dataset) for i in range(POP_SIZE)]
        if max(fitnesses) > best_of_run_f:
            best_of_run_f = max(fitnesses)
            best_of_run_gen = gen
            best_of_run = deepcopy(population[fitnesses.index(max(fitnesses))])
            print("________________________")
            print("gen:", gen, ", best_of_run_f:", round(max(fitnesses),3), ", best_of_run:") 
            best_of_run.print_tree()
        if best_of_run_f == 1: break   
        # plt.show()
    
    print("\n\n_________________________________________________\nEND OF RUN\nbest_of_run attained at gen " + str(best_of_run_gen) +\
          " and has f=" + str(round(best_of_run_f,3)))
    best_of_run.print_tree()
    
if __name__== "__main__":
  main()