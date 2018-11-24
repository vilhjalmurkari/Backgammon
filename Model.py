
import torch
from torch.autograd import Variable
import agent

class Model:
    def __init__(self,_player,useTrained,loadtrainstep,_lambda,_alpha):
        self.device = torch.device('cpu') #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if useTrained:
            #self.w1 = torch.load('./trainedWeights/w1_trained_'+str(loadtrainstep)+'.pth')
            #self.w2 = torch.load('./trainedWeights/w2_trained_'+str(loadtrainstep)+'.pth')
            #self.b1 = torch.load('./trainedWeights/b1_trained_'+str(loadtrainstep)+'.pth')
            #self.b2 = torch.load('./trainedWeights/b2_trained_'+str(loadtrainstep)+'.pth')
            #self.w1 = torch.load('./trainedWeights/w1_trained_'+'-'+str(loadtrainstep)+'_player_'+str(_player)+'-'+'lam'+str(_lambda)+'alpha'+str(_alpha)+'.pth')
            #self.w2 = torch.load('./trainedWeights/w2_trained_'+'-'+str(loadtrainstep)+'_player_'+str(_player)+'-'+'lam'+str(_lambda)+'alpha'+str(_alpha)+'.pth')
            #self.b1 = torch.load('./trainedWeights/b1_trained_'+'-'+str(loadtrainstep)+'_player_'+str(_player)+'-'+'lam'+str(_lambda)+'alpha'+str(_alpha)+'.pth')
            #self.b2 = torch.load('./trainedWeights/b2_trained_'+'-'+str(loadtrainstep)+'_player_'+str(_player)+'-'+'lam'+str(_lambda)+'alpha'+str(_alpha)+'.pth')
        else:
            self.w1 = Variable(0.001*torch.randn(40,28*31, device = self.device, dtype=torch.float), requires_grad = True)
            self.b1 = Variable(torch.zeros((40,1), device = self.device, dtype=torch.float), requires_grad = True)
            self.w2 = Variable(0.001*torch.randn(1, 40, device = self.device, dtype=torch.float), requires_grad = True)
            self.b2 = Variable(torch.zeros((1,1), device = self.device, dtype=torch.float), requires_grad = True)
        self.gamesWon = 0
        self.player = _player
        self.alpha1 = _alpha
        self.alpha2 = _alpha
        self.lam = _lambda
        self.xold = Variable(torch.tensor(torch.zeros(868,1), dtype=torch.float, device = self.device)).view(28*31,1)
        self.Z_w1 = torch.zeros(self.w1.size(), device = self.device, dtype = torch.float)
        self.Z_b1 = torch.zeros(self.b1.size(), device = self.device, dtype = torch.float)
        self.Z_w2 = torch.zeros(self.w2.size(), device = self.device, dtype = torch.float)
        self.Z_b2 = torch.zeros(self.b2.size(), device = self.device, dtype = torch.float)
        self.theta = 0.01*torch.ones((1,40), device=self.device,dtype = torch.float)
        self.alpha_th = 0.0001
        self.xtheta = 0.0001

    def gameFinishedUpdate(self,winner):
        reward = 1 if winner == self.player else 0
        if(reward==1):
            self.gamesWon += 1
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

        # perform now the update of weights
        delta2 =  torch.tensor(delta2, dtype = torch.float, device = self.device)

        # zero the gradients
        self.w2.grad.data.zero_()
        self.b2.grad.data.zero_()
        self.w1.grad.data.zero_()
        self.b1.grad.data.zero_()

        self.w1.data = self.w1.data + self.alpha1 * delta2 * self.Z_w1
        self.b1.data = self.b1.data + self.alpha1 * delta2 * self.Z_b1
        self.w2.data = self.w2.data + self.alpha2 * delta2 * self.Z_w2
        self.b2.data = self.b2.data + self.alpha2 * delta2 * self.Z_b2

        grad_ln_pi = h_sigmoid - self.xtheta
        self.theta.data = self.theta.data + self.alpha_th*delta2*grad_ln_pi
        

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

        # perform now the update for the weights
        delta2 =  torch.tensor(delta2, dtype = torch.float, device = self.device)
        
        # zero the gradients
        self.w2.grad.data.zero_()
        self.b2.grad.data.zero_()
        self.w1.grad.data.zero_()
        self.b1.grad.data.zero_()

        self.w1.data = self.w1.data + self.alpha1 * delta2 * self.Z_w1
        self.b1.data = self.b1.data + self.alpha1 * delta2 * self.Z_b1
        self.w2.data = self.w2.data + self.alpha2 * delta2 * self.Z_w2
        self.b2.data = self.b2.data + self.alpha2 * delta2 * self.Z_b2

        grad_ln_pi = h_sigmoid - self.xtheta
        self.theta.data = self.theta.data + self.alpha_th*delta2*grad_ln_pi
        

    def saveNetwork(self,totalTrained):
        torch.save(self.w1, /.sn       i8y64tv5tt6cgt''./trainedWeights/w1_trained_'+'-'+str(loadtrainstep)+'_player_'+str(_player)+'-'+'lam'+str(_lambda)+'alpha'+str(_alpha)+'.pth'+str(self.player)+'-'+'lam'+str(self.lam)+'alpha'+str(self.alpha1)+'.pth')
        torch.save(self.w2, './trainedWeights/w2_trained_'+str(totalTrained)+'_player_'+str(self.player)+'-'+'lam'+str(self.lam)+'alpha'+str(self.alpha1)+'.pth')
        torch.save(self.b1, './trainedWeights/b1_trained_'+str(totalTrained)+'_player_'+str(self.player)+'-'+'lam'+str(self.lam)+'alpha'+str(self.alpha1)+'.pth')
        torch.save(self.b2, './trainedWeights/b2_trained_'+str(totalTrained)+'_player_'+str(self.player)+'-'+'lam'+str(self.lam)+'alpha'+str(self.alpha1)+'.pth')
    