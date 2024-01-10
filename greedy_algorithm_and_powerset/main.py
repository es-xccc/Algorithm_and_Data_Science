import time

itemlist = []


class items:
    def __init__(self, a, b, c, d):
        self.name = a
        self.price = b
        self.oriprice = c
        self.discount = d

    def __lt__(self, other):  # greedy
        return self.discount / self.price < other.discount / other.price

    def __str__(self):
        result = '<' + self.name + ', ' + 'price:' + str(self.discount) + ', ' + 'original price:' + \
                 str(self.oriprice) + ', ' + 'discount:' + str(self.price) + '>'
        return result


def buildItems():
    names = ['紳士鞋', '平板保護套', '有線耳機', '列表機', '太陽眼鏡', '炭板跑鞋', '慢跑鞋', '運動手錶', '無限喇叭',
             '運動服', '足球運動套裝', '運動鞋', '筆電帶', '藍芽耳機', '滑板', '泳衣', '遊戲機', '水族箱', '自行車頭盔',
             '吹風機', '床單', '行動電源']

    prices = [878, 1140, 2280, 760, 1330, 2099, 2508, 2850, 3762, 1539, 3190, 2153, 3724, 2964, 1406, 1214, 1482, 1060,
              1140, 1482, 1209, 1518]

    oriprices = [1406, 1900, 5700, 2280, 3420, 3418, 3230, 3487, 7562, 2052, 3420, 2658, 4331, 4940, 2280, 2052, 2470,
                 1250, 1290, 2280, 1512, 1786]

    discounts = [528, 760, 3420, 1520, 2090, 1319, 722, 637, 3800, 513, 230, 505, 607, 1976, 874, 838, 988, 190, 151,
                 798, 302, 268]
    for i in range(len(names)):
        itemlist.append(items(names[i], prices[i], oriprices[i], discounts[i]))


def greedy(itemlist1, maxPri):
    """Assumes Items a list, maxWeight >= 0,
        keyFunction maps elements of Items to numbers"""
    itemsCopy = sorted(itemlist1, reverse=True)
    result = []
    total_discount, total_price = 0, 0
    for i in range(len(itemsCopy)):
        if (total_price + itemsCopy[i].price) <= maxPri:
            result.append(itemsCopy[i])
            total_price += itemsCopy[i].price
            total_discount += itemsCopy[i].discount
    return result, total_discount, total_price


def testGreedy(items1, max_price):
    print('Greedy 使用 discount / price')
    taken, dis_sum, pri_sum = greedy(items1, max_price)
    print('Total number of items taken is', len(taken))
    for item in taken:
        print(' ', item)
    print('Total price of items taken is', pri_sum)
    print('Total discount of items taken is', dis_sum)


def chooseBest(pset, maxPri):
    bestDis = 0
    bestPri = 0
    bestSet = None
    for items in pset:
        itemsDis = 0
        itemsPri = 0
        for item in items:
            itemsDis += item.discount
            itemsPri += item.price
        if itemsPri <= maxPri and itemsDis > bestDis:
            bestDis = itemsDis
            bestPri = itemsPri
            bestSet = items
    return bestSet, bestDis, bestPri


def getBinaryRep(n, numDigits):
    """Assumes n and numDigits are non-negative ints
    Returns a str of length numDigits that is a binary representation of n"""
    result = ''
    while n > 0:
        result = str(n % 2) + result
        n = n // 2
    if len(result) > numDigits:
        raise ValueError('not enough digits')
    for i in range(numDigits - len(result)):
        result = '0' + result
    return result


def genPowerset(L):
    """Assumes L is a list Returns a list of lists that contains
    all possible combinations of the elements of L. E.g.,
    if L is [1, 2] it will return a list with elements [], [1], [2], and [1,2]."""
    powerset = []
    for i in range(0, 2 ** len(L)):
        binStr = getBinaryRep(i, len(L))
        subset = []
        for j in range(len(L)):
            if binStr[j] == '1':
                subset.append(L[j])
        powerset.append(subset)
    return powerset


def testBest(items1, max_price):
    print('Optimal 使用 Powerset')
    pset = genPowerset(items1)
    taken, dis_sum, pri_sum = chooseBest(pset, max_price)
    print('Total number of items taken is', len(taken))
    for item in taken:
        print(item)
    print('Total price of items taken is', pri_sum)
    print('Total discount of items taken is', dis_sum)


buildItems()
greedy_st = time.time()
testGreedy(itemlist, 30000)
greedy_et = time.time()
print("greedy cost time:{:.10f}".format(greedy_et - greedy_st))
print('----------------------------------------')
best_st = time.time()
testBest(itemlist, 30000)
best_et = time.time()
print("optimal cost time:{:.10f}".format(best_et - best_st))
