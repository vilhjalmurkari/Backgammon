myDict = {}
array = [1,2,3,4]
myDict[str(array)] = 10
myDict["[0,0,0]"] = 199
print(myDict["[1,2,3,4]"])
print(myDict["[0,0,0]"])
myDict["[0,0,0]"] += 10
print(myDict["[0,0,0]"])
print(myDict[str(array)])