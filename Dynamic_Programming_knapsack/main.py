import random
import numpy as np
import time

def buildManyItems(numItems, maxVal, maxWeight):  
    v = [0, 8, 8, 1, 18, 14, 4, 16, 1]
    w = [0, 3, 2, 7, 19, 3, 10, 1, 14]
    # for i in range(numItems): 
    #     v.append(random.randint(1, maxVal))
    #     w.append(random.randint(1, maxWeight))
    return v, w

def knapsack(i, j):
    if i == 0 :
        return {}
    if items[i][j] > items[i-1][j]:
        return {i}.union(knapsack(i-1, j-weights[i]))
    else:
        return knapsack(i-1, j)
    
def bigTest1(numItems, capacity): 
    global values, weights    
    values, weights = buildManyItems(numItems, 20, 20) 
    print('values =', values[1:])
    print('weights =', weights[1:])
    print('---------------------------------------------------')
    global items
    items = np.zeros((numItems+1, capacity+1), dtype=int)  #create a numItems*maxWeight array with elements initialized to 0

    for i in range(1, numItems+1):    # Cponsider every objects
        for j in range(capacity+1): # Condsider every object's weight
            if weights[i] > j : # wi>W the new item is more than the current weight limit 
                items[i][j] = items[i-1][j]
            else :                # wi<=W ????
                items[i][j] = max(items[i-1][j], items[i-1][j-weights[i]] + values[i])

    print(items)    
    print('items', knapsack(numItems,capacity), 'selected.')
    return items[numItems][capacity]

def bigTest2(numItems, capacity):
    items = np.full((numItems+1, capacity+1), -1, dtype=int)
    def m(i, j):
        if i == 0 or j <= 0:
            items[i, j] = 0
            return
        if items[i-1, j] == -1:
            m(i-1, j)
        if weights[i] > j:
            items[i, j] = items[i-1, j]
        else:
            if items[i-1, j-weights[i]] == -1:
                m(i-1, j-weights[i])
            items[i, j] = max(items[i-1, j], items[i-1, j-weights[i]] + values[i])
    m(numItems, capacity)
    print(items)    
    print('items', knapsack(numItems,capacity), 'selected.')
    return items[numItems][capacity]


start1 = time.time()
print('Total value selected =', bigTest1(8, 20))
end1 = time.time()
print('Execution time of non recursive version is:', end1 - start1)
print('---------------------------------------------------')

start2 = time.time()
print('Total value selected =', bigTest2(8, 20))
end2 = time.time()
print('Execution time of recursive version is:', end2 - start2)
