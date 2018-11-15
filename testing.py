import TestClass

def breytaAftur(model):
    model.a = 5000
def breyta(model):
    breytaAftur(model)

a = 100
testClass = TestClass.Test()
testClass.a = a 
print(testClass.a)   
breyta(testClass)
print(testClass.a)