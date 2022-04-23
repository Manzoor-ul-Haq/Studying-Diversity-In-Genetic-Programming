# tiny genetic programming by © moshe sipper, www.moshesipper.com
from random import random, randint, seed
import random as rdm
from statistics import mean
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np
from os import path
import seaborn as sns
from functools import reduce, partial
import pandas as pd
from openpyxl import load_workbook
import xlwt
from xlwt import Workbook

POP_SIZE        = 60   # population size
MIN_DEPTH       = 2    # minimal initial random tree depth
MAX_DEPTH       = 5    # maximal initial random tree depth
GENERATIONS     = 500  # maximal number of generations to run evolution
TOURNAMENT_SIZE = 5    # size of tournament for tournament selection
XO_RATE         = 0.8  # crossover rate
PROB_MUTATION   = 0.2  # per-node mutation probability
PRUNE_RATE      = 0.3  # pruning rate

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

def protectedDiv(left, right):
    with np.errstate(divide='ignore',invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x

class GPTree:
    postOrderedList = []
    left1 = 1
    right1 = 1
    data1 = 1
    counter = 0
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

    # def hash(self, node):
    #     if not node:
    #         return 1

    #     if node.left:
    #         self.left1 = -2

    #     if node.right:
    #         self.right1 = -1
        
    #     if node:
    #         if node.data == add:
    #             self.data1 = 2
    #         elif node.data == mul:
    #             self.data1 = 0.5
    #         elif node.data == sub:
    #             self.data1 = -3
    #         elif node.data == 'x':
    #             self.data1 = 0.25
    #         elif node.data == 0:
    #             self.data1 = 1
    #         else:
    #             self.data1 = node.data

    #     return self.left1 * self.right1 * self.data1 * self.hash(node.left) * self.hash(node.right)

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
        if subtreesTree1 == subtreesTree2:
            return subtreesTree1
        
        union = []

        temp1 = subtreesTree1
        temp2 = subtreesTree2

        i, j = 0, 0

        while(i < len(temp1)):
            j = 0
            while(j < len(temp2)):
                if temp1[i] == temp2[j]:
                    t1 = temp1.pop(temp1.index(temp1[i]))
                    t2 = temp2.pop(temp2.index(temp2[j]))
                    union.append(t1)
                    i = -1
                    break
                j += 1
            i += 1

        for i in range(len(temp1)):
            union.append(temp1[i])

        for i in range(len(temp2)):
            union.append(temp2[i])
        
        return union

    def intersection(self, subtreesTree1, subtreesTree2):
        if subtreesTree1 == subtreesTree2:
            return []
        
        intersection = []

        for i in range(len(subtreesTree1)):
            if subtreesTree1[i] in subtreesTree2:
                intersection.append(subtreesTree1[i])

        return intersection

    def jaccardIndex(self, tree1, tree2):
        subtreesTree1 = self.tuplesSubtree(tree1)
        subtreesTree2 = self.tuplesSubtree(tree2)

        intersection = self.intersection(subtreesTree1, subtreesTree2)
        if intersection == []:
            return 1
        
        union = self.union(subtreesTree1, subtreesTree2)
        if union == []:
            return 0

        return len(intersection) / len(union)

    def jaccard_similarityMatrix(self, list1):
        matrix = np.zeros((len(list1), len(list1)))
        
        list2 = deepcopy(list1)

        for i in range(len(list1)):
            for j in range(len(list2)):
                matrix[i][j] = -1 * self.jaccardIndex(list1[i], list2[j])
                matrix[i][j] += 1

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
        
        list2 = deepcopy(list1)

        for i in range(len(list1)):
            for j in range(len(list2)):
                matrix[i][j] = self.treedist(list1[i], list2[j])

        # print(matrix.max())
        # print(matrix.min())

        for i in range(len(list1)):
            for j in range(len(list2)):
                matrix[i][j] = protectedDiv(matrix[i][j], matrix.max())
                # matrix[i][j] += 1
        
        # print(matrix[20][19])
        # sys.exit()
        # print(matrix.max())
        # print(matrix.min())
        # print(matrix.mean())
        # sys.exit()
        return matrix

# end class GPTree
                   
def init_population(): # ramped half-and-half
    pop = []
    # for md in range(3, MAX_DEPTH + 1):
    for i in range(POP_SIZE):
        t = GPTree()
        # t.random_tree(grow = True, max_depth = md) # grow
        t.random_tree(grow = bool(rdm.getrandbits(1)), max_depth = randint(3, MAX_DEPTH)) # grow
        pop.append(t)
        # for i in range(int(POP_SIZE/5)):
        #     t = GPTree()
        #     # t.random_tree(grow = False, max_depth = md) # full
        #     t.random_tree(grow = bool(rdm.getrandbits(1)), max_depth = randint(3, MAX_DEPTH)) # full
        #     pop.append(t)
    return pop

def fitness(individual, dataset): # inverse mean absolute error over dataset normalized to [0,1]
    return 1 / (1 + mean([abs(individual.compute_tree(ds[0]) - ds[1]) for ds in dataset]))
                
def selection(population, fitnesses): # select one individual using tournament selection
    tournament = [randint(0, len(population)-1) for i in range(TOURNAMENT_SIZE)] # select tournament contenders
    tournament_fitnesses = [fitnesses[tournament[i]] for i in range(TOURNAMENT_SIZE)]
    return deepcopy(population[tournament[tournament_fitnesses.index(max(tournament_fitnesses))]])   

def generation_wideSimplification(population):
    subTrees = []
    Tree = GPTree()
    man, zoor, dataset = generate_dataset()
    hashValues = []

    fitnesses = [fitness(population[i], dataset) for i in range(POP_SIZE)] 

    for i in range(len(population)):
        subTrees.extend(Tree.childrenPostOrderedList(population[i]))

    subTrees1 = deepcopy(subTrees)

    for i in range(len(subTrees)):
        hashValues.append(hash(subTrees[i]))

    hashValues1 = deepcopy(hashValues)

    for i in range(len(hashValues1)):
        j = 0
        while j < len(hashValues):
            if hashValues1[i] == hashValues[j]:
                hashValues.pop(j)
                subTrees.pop(j)
                j -= 1
            j += 1

    fitnesses_subTrees = [fitness(subTrees[i], dataset) for i in range(len(subTrees))]
    sortedFitnesses = deepcopy(fitnesses_subTrees)
    sortedFitnesses.sort(reverse=True)
    sorted_subTrees = [subTrees[fitnesses_subTrees.index(sortedFitnesses[i])] for i in range(len(subTrees))]
    if len(sorted_subTrees) < POP_SIZE:
        for i in range(POP_SIZE - len(sorted_subTrees)):
            parent1 = selection(population, fitnesses)
            parent2 = selection(population, fitnesses)
            parent1.crossover(parent2)
            parent1.mutation()
            sorted_subTrees.append(parent1)
        return sorted_subTrees
    else:
       return sorted_subTrees[:POP_SIZE]

def pruning(population):
    Tree = GPTree()
    man, zoor, dataset = generate_dataset()
    no_of_individuals_for_pruning = PRUNE_RATE * len(population)
    fitnesses = [fitness(population[i], dataset) for i in range(POP_SIZE)]
    sortedFitnesses = deepcopy(fitnesses)
    sortedFitnesses.sort(reverse=True)
    population = [population[fitnesses.index(sortedFitnesses[i])] for i in range(POP_SIZE)]
    for i in range(int(no_of_individuals_for_pruning)):
        subTrees = Tree.childrenPostOrderedList(population[i])
        fitness_of_subtrees = [fitness(subTrees[j], dataset) for j in range(len(subTrees))]
        max_fitness_subtree = max(fitness_of_subtrees)
        index = fitness_of_subtrees.index(max_fitness_subtree)
        if max_fitness_subtree > fitness(population[i], dataset):
            population[i] = subTrees[index]
    return population

def averageProgramSize(population):
    total = 0
    for i in population:
        total += i.size()
    return total/len(population)

def maxProgramSize(population):
    total = 0
    for i in population:
        if i.size() > total:
            total = i.size()
    return total


def main():
    # init stuff
    seed() # init internal state of random number generator
    man, zoor, dataset = generate_dataset()

    population = init_population()
    # population = pruning(population)

    # newPopulation = generation_wideSimplification(population)
    t = GPTree()
    max_fitnesses = []
    avg_fitnesses = []
    sumTed = []
    sumJaccard = []
    
    fitnesses = [fitness(population[i], dataset) for i in range(POP_SIZE)]
    sortedFitnesses = deepcopy(fitnesses)
    sortedFitnesses.sort(reverse=True)
    print("best: ", sortedFitnesses[0], max(fitnesses))
    populate = [population[fitnesses.index(sortedFitnesses[i])] for i in range(POP_SIZE)]
    
    ted = t.ted_similarityMatrix(populate)
    result = ted.flatten()
    tedSum = sum(result)
    sumTed.append(tedSum)

    jaccard = t.jaccard_similarityMatrix(populate)
    result = jaccard.flatten()
    jaccardSum = sum(result)
    sumJaccard.append(jaccardSum)

    # columns = []
    # for i in range(60):
    #     columns.append("T" + str(i+1))
    # print(columns)


    # graphs
    
    outpath = "C:/Users/admin/Documents/Namal/Fall 2021/CSE-491 Final Year Project-1/Studying-Diversity-In-Genetic-Programming/Gws25_TED"
    
    ax = sns.heatmap(ted, linewidth=0.5, vmin=0, vmax=1)
    ax.set_title("Generation 0 (Zhang & Shasha)")
    plt.savefig(path.join(outpath,"Generation_0.png"))
    plt.close('all')
    
    outpath = "C:/Users/admin/Documents/Namal/Fall 2021/CSE-491 Final Year Project-1/Studying-Diversity-In-Genetic-Programming/Gws25_Jaccard"

    ax = sns.heatmap(jaccard, linewidth=0.5, vmin=0, vmax=1)
    ax.set_title("Generation 0 (Jaccard Index)")
    plt.savefig(path.join(outpath,"Generation_0.png"))
    plt.close('all')

    # sys.exit()

    best_of_run = None
    best_of_run_f = 0
    best_of_run_gen = 0
    max_fitnesses.append(max(fitnesses))
    avg_fitnesses.append(mean(fitnesses))
    # sys.exit()
    # counter = 0
    # go evolution!
    wb = Workbook()
    sheet1 = wb.add_sheet('Gws25_Gws25')
    sheet1.write(0, 0, 'Generations')
    sheet1.write(0, 1, 'Average Fitness')
    sheet1.write(0, 2, 'Maximum Fitness')
    sheet1.write(0, 3, 'Jaccard Diversity')
    sheet1.write(0, 4, '')
    sheet1.write(0, 5, 'TED Diversity')
    sheet1.write(0, 6, '')
    sheet1.write(0, 7, 'Average Program Size')
    sheet1.write(0, 8, 'Maximum Program Size')
    
    sheet1.write(1, 0, 0)
    sheet1.write(1, 1, mean(fitnesses))
    sheet1.write(1, 2, sortedFitnesses[0])
    sheet1.write(1, 3, jaccardSum)
    sheet1.write(1, 4, protectedDiv(jaccardSum, 3600))
    sheet1.write(1, 5, tedSum)
    sheet1.write(1, 6, protectedDiv(tedSum, 3600))
    sheet1.write(1, 7, averageProgramSize(population))
    sheet1.write(1, 8, maxProgramSize(population))

    for gen in range(GENERATIONS):  
        print(gen)
        nextgen_population=[]
        if (gen+1) % 25 == 0:
            nextgen_population = generation_wideSimplification(population)
            population = nextgen_population
        else:
            for i in range(POP_SIZE):
                parent1 = selection(population, fitnesses)
                parent2 = selection(population, fitnesses)
                parent1.crossover(parent2)
                parent1.mutation()
                nextgen_population.append(parent1)
            population = nextgen_population
            
        # population = pruning(population)

        # print("POP_SIZE", len(population))
        
        fitnesses = [fitness(population[i], dataset) for i in range(POP_SIZE)]
        sortedFitnesses = fitnesses
        sortedFitnesses.sort(reverse=True)
        populate = [population[fitnesses.index(sortedFitnesses[i])] for i in range(POP_SIZE)]
        print("fit", sortedFitnesses[0])

        # counter += 1
        ted = t.ted_similarityMatrix(populate)
        result = ted.flatten()
        tedSum = sum(result)
        sumTed.append(tedSum)

        jaccard = t.jaccard_similarityMatrix(populate)
        result = jaccard.flatten()
        jaccardSum = sum(result)
        sumJaccard.append(jaccardSum)
    
        # df = pd.DataFrame({'Generations': [gen+1], 'Average Fitness': [mean(fitnesses)], 'Maximum Fitness': [sortedFitnesses[0]],
        #     'Jaccard Diversity': [1/jaccardSum], 'TED Diversity': [1/tedSum], 'Average Program Size': [averageProgramSize(population)],
        #     'Maximum Program Size': [maxProgramSize(population)]})
        sheet1.write(gen+2, 0, gen+1)
        sheet1.write(gen+2, 1, mean(fitnesses))
        sheet1.write(gen+2, 2, sortedFitnesses[0])
        sheet1.write(gen+2, 3, jaccardSum)
        sheet1.write(gen+2, 4, protectedDiv(jaccardSum, 3600))
        sheet1.write(gen+2, 5, tedSum)
        sheet1.write(gen+2, 6, protectedDiv(tedSum, 3600))
        sheet1.write(gen+2, 7, averageProgramSize(population))
        sheet1.write(gen+2, 8, maxProgramSize(population))

        #Graphs
        if gen == 24 or gen == 49 or gen == 99 or gen == 124:
            print(jaccard.max(), "max")
            print(jaccard.min(), "min")
            print(jaccard.mean(), "mean")

        outpath = "C:/Users/admin/Documents/Namal/Fall 2021/CSE-491 Final Year Project-1/Studying-Diversity-In-Genetic-Programming/Gws25_TED"
        
        ax = sns.heatmap(ted, linewidth=0.5, vmin=0, vmax=1)
        ax.set_title("Generation " + str(gen+1) + " (Zhang & Shasha)")
        plt.savefig(path.join(outpath,"Generation_" + str(gen+1) + ".png"))
        plt.close('all')

        outpath = "C:/Users/admin/Documents/Namal/Fall 2021/CSE-491 Final Year Project-1/Studying-Diversity-In-Genetic-Programming/Gws25_Jaccard"

        ax = sns.heatmap(jaccard, linewidth=0.5, vmin=0, vmax=1)
        ax.set_title("Generation " + str(gen+1) + " (Jaccard Index)")
        plt.savefig(path.join(outpath,"Generation_" + str(gen+1) + ".png"))
        plt.close('all')
        
        max_fitnesses.append(max(fitnesses))
        avg_fitnesses.append(mean(fitnesses))
        if max(fitnesses) > best_of_run_f:
            best_of_run_f = max(fitnesses)
            best_of_run_gen = gen
            best_of_run = deepcopy(population[fitnesses.index(max(fitnesses))])
            print("________________________")
            print("gen:", gen, ", best_of_run_f:", round(max(fitnesses), 3), ", best_of_run:") 
            # best_of_run.print_tree()
        if best_of_run_f == 1: break   
        # plt.show()
    
    wb.save('Gws25_Gws25.xls')
    print("\n\n_________________________________________________\nEND OF RUN\nbest_of_run attained at gen " + str(best_of_run_gen) +\
          " and has f=" + str(round(best_of_run_f,3)))
    # best_of_run.print_tree()

    outpath = "C:/Users/admin/Documents/Namal/Fall 2021/CSE-491 Final Year Project-1/Studying-Diversity-In-Genetic-Programming/Gws25_Graphs"
    x = np.arange(0, len(max_fitnesses))
    y = np.array(max_fitnesses)
     
    # plotting
    plt.title("Max Fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.plot(x, y, color ="green")
    plt.savefig(path.join(outpath,"MaxFitness.png"))
    plt.close()
    
    x = np.arange(0, len(avg_fitnesses))
    y = np.array(avg_fitnesses) 
    # plotting
    plt.title("Average Fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.plot(x, y, color ="red")
    plt.savefig(path.join(outpath,"AvgFitness.png"))
    plt.close()
    
    x = np.arange(0, len(sumTed))
    y = np.array(sumTed) 
    # plotting
    plt.title("Zhand & Shasha Similarity")
    plt.xlabel("Generations")
    plt.ylabel("Similarity Sum")
    plt.plot(x, y, color ="yellow")
    plt.savefig(path.join(outpath,"Z&SFitness.png"))
    plt.close()
    
    x = np.arange(0, len(sumJaccard))
    y = np.array(sumJaccard)
    # plotting
    plt.title("Jaccard Similarity")
    plt.xlabel("Generations")
    plt.ylabel("Similarity Sum")
    plt.plot(x, y, color ="black")
    plt.savefig(path.join(outpath,"JSFitness.png"))
    plt.close()
    
if __name__== "__main__":
  main()

  # exp(-x) * x**3 * cos(x) * sin(x) * (cos(x) * sin(x)**2 - 1)