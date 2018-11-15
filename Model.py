
import torch
from torch.autograd import Variable
import agent

class Model:
    device = torch.device('cpu') #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    w1 = Variable(torch.randn(28*28,28*31, device = device, dtype=torch.float), requires_grad = True)
    b1 = Variable(torch.zeros((28*28,1), device = device, dtype=torch.float), requires_grad = True)
    w2 = Variable(torch.randn(1, 28*28, device = device, dtype=torch.float), requires_grad = True)
    b2 = Variable(torch.zeros((1,1), device = device, dtype=torch.float), requires_grad = True)
    alpha1 = 0.2
    alpha2 = 0.2
    lam = 0.4
    xold = Variable(torch.tensor(torch.zeros(868,1), dtype=torch.float, device = device)).view((28*31,1))
    Z_w1 = torch.zeros(w1.size(), device = device, dtype = torch.float)
    Z_b1 = torch.zeros(b1.size(), device = device, dtype = torch.float)
    Z_w2 = torch.zeros(w2.size(), device = device, dtype = torch.float)
    Z_b2 = torch.zeros(b2.size(), device = device, dtype = torch.float)


    def gameFinishedUpdate(self,winner):
        reward = 1 if winner == 1 else 0
        gamma = 1
        h = torch.mm(self.w1,self.xold) + self.b1 # matrix-multiply x with input weight w1 and add bias
        h_sigmoid = h.sigmoid() # squash this with a sigmoid function
        y = torch.mm(self.w2,h_sigmoid) + self.b2 # multiply with the output weights w2 and add bias
        y_sigmoid = y.sigmoid() # squash the output
        delta2 = reward + gamma * 0 - y_sigmoid.detach().cpu().numpy()  # this is the usual TD error
        # using autograd and the contructed computational graph in pytorch compute all gradients
        y_sigmoid.backward()
        # update the eligibility traces
        self.Z_w2 = gamma * self.lam * self.Z_w2 + self.w2.grad.data
        self.Z_b2 = gamma * self.lam * self.Z_b2 + self.b2.grad.data
        self.Z_w1 = gamma * self.lam * self.Z_w1 + self.w1.grad.data
        self.Z_b1 = gamma * self.lam * self.Z_b1 + self.b1.grad.data
            
        import time 
        time.sleep(10)
        # zero the gradients
        self.w2.grad.data.zero_()
        self.b2.grad.data.zero_()
        self.w1.grad.data.zero_()
        self.b1.grad.data.zero_()

        # perform now the update of weights
        delta2 =  torch.tensor(delta2, dtype = torch.float, device = self.device)

        self.w1.data = self.w1.data + self.alpha1 * delta2 * self.Z_w1
        self.b1.data = self.b1.data + self.alpha1 * delta2 * self.Z_b1
        self.w2.data = self.w2.data + self.alpha2 * delta2 * self.Z_w2
        self.b2.data = self.b2.data + self.alpha2 * delta2 * self.Z_b2

    def updateNeural(self,after_state):
        gamma=1
        # here we have player 2 updating the neural-network (2 layer feed forward with Sigmoid units)
        x = Variable(torch.tensor(agent.one_hot_encoding(after_state), dtype = torch.float, device = self.device)).view(28*31,1)
        # now do a forward pass to evaluate the new board's after-state value
        
        h = torch.mm(self.w1,x) + self.b1 # matrix-multiply x with input weight w1 and add bias
        h_sigmoid = h.sigmoid() # squash this with a sigmoid function
        y = torch.mm(self.w2,h_sigmoid) + self.b2 # multiply with the output weights w2 and add bias
        y_sigmoid = y.sigmoid() # squash this with a sigmoid function
        target = y_sigmoid.detach().cpu().numpy()
        # lets also do a forward past for the old board, this is the state we will update
        h = torch.mm(self.w1,self.xold) + self.b1 # matrix-multiply x with input weight w1 and add bias
        h_sigmoid = h.sigmoid() # squash this with a sigmoid function
        y = torch.mm(self.w2,h_sigmoid) + self.b2 # multiply with the output weights w2 and add bias
        y_sigmoid = y.sigmoid() # squash the output
        
        delta2 = 0 + gamma*target - y_sigmoid.detach().cpu().numpy()
        # using autograd and the contructed computational graph in pytorch compute all gradients
        y_sigmoid.backward()
        # update the eligibility traces using the gradients
        self.Z_w2 = gamma * self.lam * self.Z_w2 + self.w2.grad.data
        self.Z_b2 = gamma * self.lam * self.Z_b2 + self.b2.grad.data
        self.Z_w1 = gamma * self.lam * self.Z_w1 + self.w1.grad.data
        self.Z_b1 = gamma * self.lam * self.Z_b1 + self.b1.grad.data
        # zero the gradients
        self.w2.grad.data.zero_()
        self.b2.grad.data.zero_()
        self.w1.grad.data.zero_()
        self.b1.grad.data.zero_()
        # perform now the update for the weights
        delta2 =  torch.tensor(delta2, dtype = torch.float, device = self.device)
        self.w1.data = self.w1.data + self.alpha1 * delta2 * self.Z_w1
        self.b1.data = self.b1.data + self.alpha1 * delta2 * self.Z_b1
        self.w2.data = self.w2.data + self.alpha2 * delta2 * self.Z_w2
        self.b2.data = self.b2.data + self.alpha2 * delta2 * self.Z_b2
        