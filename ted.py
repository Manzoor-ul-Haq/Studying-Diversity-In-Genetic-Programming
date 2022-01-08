import numpy as np

class Node(object):
	"""docstring for Node"""
	def __init__(self, value):
		self.left = None
		self.right = None
		self.parent = None
		self.value = value

	def inorder(node):
		if node:
			inorder(node.left)
			print(node.value)
			inorder(node.right)

class Tree():
	def __init__(self, node):
		self.node = node
		self.keyrootsList = []
		self.postOrderedList = []
		if not node.parent:
			self.keyrootsList.append(node)

	def rightRecursion(self, node):
		if node:
			self.rightRecursion(node.right)
			self.leftRecursion(node.left)
			self.keyrootsList.append(node)

	def leftRecursion(self, node):
		if node:
			self.leftRecursion(node.left)
			self.rightRecursion(node.right)

	def leftmostChild(self, node):
		while True:
			if not node.left:
				return node
			node = node.left

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

	def keyRoots(self, node):
		self.leftRecursion(node)
		array = self.childrenPostOrderedList(node)
		sortedKeyroots = []
		for i in range(len(array)):
			if array[i] in self.keyrootsList:
				sortedKeyroots.append(array[i])
		self.keyrootsList = []
		return sortedKeyroots

	def printDistances(self, distances, i_length, j_length):
		for t1 in range(i_length + 1):
			for t2 in range(j_length + 1):
				print(int(distances[t1][t2]), end=" ")
			print()

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
				if i_children[t1 - 1].value == j_children[t2 - 1].value:
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
	def tuplesSubtree(self, tree):
		T = self.childrenPostOrderedList(tree)
		subTrees = []

		for i in range(len(T)):
			if T[i].left and T[i].right:
				subTrees.append((T[i].left.value, T[i].right.value, T[i].value))
			elif T[i].left:
				subTrees.append((T[i].left.value, T[i].value))
			elif T[i].right:
				subTrees.append((T[i].right.value, T[i].value))

		print(subTrees)
		return subTrees

	def union(self, subtreesTree1, subtreesTree2):
		return list(set(subtreesTree1) | set(subtreesTree2))

	def intersection(self, subtreesTree1, subtreesTree2):
		return [i for i in subtreesTree1 if i in subtreesTree2]

	def jaccardIndex(self, tree1, tree2):
		subtreesTree1 = self.tuplesSubtree(tree1)
		subtreesTree2 = self.tuplesSubtree(tree2)

		union = self.union(subtreesTree1, subtreesTree2)
		intersection = self.intersection(subtreesTree1, subtreesTree2)

		return len(intersection) / len(union)

t1 = Node('f')
t1.left = Node('c')
t1.right = Node('e')
t1.left.left = Node('a')
t1.left.right = Node('d')
t1.left.right.left = Node('b')

t2 = Node('f')
t2.left = Node('c')
t2.right = Node('e')
t2.left.left = Node('d')
t2.left.left.left = Node('a')
t2.left.left.right = Node('b')

# tree2 = Node(10)
# tree2.parent = None
# tree2.left = Node(5)
# tree2.left.parent = tree2
# tree2.left.left = Node(3)
# tree2.left.left.parent = tree2.left
# tree2.left.right = Node(7)
# tree2.left.right.parent = tree2.left
# tree2.left.right.left = Node(6)
# tree2.left.right.left.parent = tree2.left.right
# tree2.left.right.left.right = Node(9)
# tree2.left.right.left.right.parent = tree2.left.right.left
# tree2.left.right.right = Node(8)
# tree2.left.right.right.parent = tree2.left.right
# tree2.right = Node(15)
# tree2.right.parent = tree2
# tree2.right.left = Node(14)
# tree2.right.left.parent = tree2.right
# tree2.right.left.right = Node(18)
# tree2.right.left.right.parent = tree2.right.left
# tree2.right.left.left = Node(13)
# tree2.right.left.left.parent = tree2.right.left
# tree2.right.right = Node(17)
# tree2.right.right.parent = tree2.right

# Tree2 = Tree(tree2)

# l_i = Tree2.leftmostChild(tree2)
# print(l_i.value)

# T2 = Tree2.keyRoots(tree2)

# for i in range(len(T2)):
# 	print(T2[i].value, end=", ")

# print()
# print()
# treedist = Tree2.treedist(t1.left.right, t2)
# print()
# print(treedist)

T = Tree(t1)
T1 = T.childrenPostOrderedList(t1)

T = Tree(t2)
T2 = T.childrenPostOrderedList(t2)

treedist = np.zeros((len(T1), len(T2)))

for i1 in range(len(T1)):
	for j1 in range(len(T2)):
		i = T1[i1]
		j = T2[j1]
		treedist[i1][j1] = T.treedist(i, j)

T.printDistances(treedist, len(T1) - 1, len(T2) - 1)
print()
print("Distance between T1 & T2: ", treedist[len(T1) - 1][len(T2) - 1])

print()
print(T.jaccardIndex(t1, t2))