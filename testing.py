myDict = {}
array = [1,2,3,4]
myDict[str(array)] = 10
print(myDict[str(array)])
print(len(myDict))
myDict[str(array)] = 30
print(len(myDict))
myDict[str(array)] += 10
print(len(myDict))
print(myDict[str(array)])