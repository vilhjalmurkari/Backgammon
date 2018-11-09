#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The intelligent agent
see flipped_agent for an example of how to flip the board in order to always
perceive the board as player 1
"""
import numpy as np
import Backgammon
import torch
from torch.autograd import Variable

alpha1 = 0.01
alpha2 = 0.01
epsilon = 0.1
lam = 0.4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define the parameters for the single hidden layer feed forward neural network
# randomly initialized weights with zeros for the biases
w1 = Variable(torch.randn(28*28,28*31, device = device, dtype=torch.float), requires_grad = True)
b1 = Variable(torch.zeros((28*28,1), device = device, dtype=torch.float), requires_grad = True)
w2 = Variable(torch.randn(1, 28*28, device = device, dtype=torch.float), requires_grad = True)
b2 = Variable(torch.zeros((1,1), device = device, dtype=torch.float), requires_grad = True)

Z_w1 = torch.zeros(w1.size(), device = device, dtype = torch.float)
Z_b1 = torch.zeros(b1.size(), device = device, dtype = torch.float)
Z_w2 = torch.zeros(w2.size(), device = device, dtype = torch.float)
Z_b2 = torch.zeros(b2.size(), device = device, dtype = torch.float)

xold = None
actionCount = 0
reward = 0

def initAgent():
    global Z_w2,Z_w1,Z_b1,Z_b2,xold,w1,w2,b1,b2,xold,reward,actionCount
    alpha1 = 0.01
    alpha2 = 0.01
    epsilon = 0.1
    lam = 0.4
    xold = None
    actionCount = 0
    reward = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define the parameters for the single hidden layer feed forward neural network
    # randomly initialized weights with zeros for the biases
    w1 = Variable(torch.randn(28*28,28*31, device = device, dtype=torch.float), requires_grad = True)
    b1 = Variable(torch.zeros((28*28,1), device = device, dtype=torch.float), requires_grad = True)
    w2 = Variable(torch.randn(1, 28*28, device = device, dtype=torch.float), requires_grad = True)
    b2 = Variable(torch.zeros((1,1), device = device, dtype=torch.float), requires_grad = True)

    Z_w1 = torch.zeros(w1.size(), device = device, dtype = torch.float)
    Z_b1 = torch.zeros(b1.size(), device = device, dtype = torch.float)
    Z_w2 = torch.zeros(w2.size(), device = device, dtype = torch.float)
    Z_b2 = torch.zeros(b2.size(), device = device, dtype = torch.float)


# this function is used to prepare the raw board as input to the network
# for some games (not here) it may be useful to invert the board and see it from the perspective of "player"
def one_hot_encoding(board):
    one_hot = []
    for i in range(1,len(board)):
        #create a vector with all possible quantities
        one_hot_place = np.zeros( (2 * 15) + 1 )
        
        if(board[i] == 0):    
            place_in_vector = 0
        elif (board[i] > 0):
            place_in_vector = int(board[i])
        else:
            place_in_vector = 15 + -1*int(board[i])
        
        one_hot_place[place_in_vector] = 1
        one_hot.extend(one_hot_place)
    return one_hot


def epsilon_nn_greedy(board, possible_moves, possible_boards, player):
    global epsilon
    if (np.random.uniform() < epsilon):
        #print("explorative move")
        rand = np.random.randint(len(possible_moves))
        return possible_boards[rand],possible_moves[rand]

    va = np.zeros(len(possible_moves))
    
    for i in range(0,len(possible_moves)):
        #print(i)
        x = Variable(torch.tensor(one_hot_encoding(possible_boards[i]), dtype = torch.float, device = device)).view(28*31,1)
        
        h1 = torch.mm(w1,x)
        h1 = h1.sigmoid()
        
        h2 = torch.mm(w2,h1)
        
        y = h2.sigmoid()
        #print(y_pred)
        
        va[i] = y
        
    #print(va)
    
    bestMove = np.argmax(va)
    return possible_boards[bestMove],possible_moves[bestMove]

def updateNeural(after_state,gamma, lam, alpha1, alpha2):
    global Z_w2,Z_w1,Z_b1,Z_b2,xold,w1,w2,b1,b2,epsilon
    # here we have player 2 updating the neural-network (2 layer feed forward with Sigmoid units)
    x = Variable(torch.tensor(one_hot_encoding(after_state), dtype = torch.float, device = device)).view(28*31,1)
    # now do a forward pass to evaluate the new board's after-state value
    h = torch.mm(w1,x) + b1 # matrix-multiply x with input weight w1 and add bias
    h_sigmoid = h.sigmoid() # squash this with a sigmoid function
    y = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
    y_sigmoid = y.sigmoid() # squash this with a sigmoid function
    target = y_sigmoid.detach().cpu().numpy()
    # lets also do a forward past for the old board, this is the state we will update
    h = torch.mm(w1,xold) + b1 # matrix-multiply x with input weight w1 and add bias
    h_sigmoid = h.sigmoid() # squash this with a sigmoid function
    y = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
    y_sigmoid = y.sigmoid() # squash the output
    delta2 = 0 + gamma * target - y_sigmoid.detach().cpu().numpy() # this is the usual TD error
    # using autograd and the contructed computational graph in pytorch compute all gradients
    y_sigmoid.backward()
    # update the eligibility traces using the gradients
    Z_w2 = gamma * lam * Z_w2 + w2.grad.data
    Z_b2 = gamma * lam * Z_b2 + b2.grad.data
    Z_w1 = gamma * lam * Z_w1 + w1.grad.data
    Z_b1 = gamma * lam * Z_b1 + b1.grad.data
    # zero the gradients
    w2.grad.data.zero_()
    b2.grad.data.zero_()
    w1.grad.data.zero_()
    b1.grad.data.zero_()
    # perform now the update for the weights
    delta2 =  torch.tensor(delta2, dtype = torch.float, device = device)
    w1.data = w1.data + alpha1 * delta2 * Z_w1
    b1.data = b1.data + alpha1 * delta2 * Z_b1
    w2.data = w2.data + alpha2 * delta2 * Z_w2
    b2.data = b2.data + alpha2 * delta2 * Z_b2

def action(board_copy,dice,player,i):
    global actionCount
    gamma = 1
    # check out the legal moves available for the throw
    possible_moves, possible_boards = Backgammon.legal_moves(board_copy, dice, player)
    
    # if there are no moves available
    if len(possible_moves) == 0: 
        return [] 
    
    #Backgammon.pretty_print(board_copy)
    #print(len(possible_boards))

    after_state,action = epsilon_nn_greedy(board_copy, possible_moves, possible_boards, player)
    
    global xold
    xold = Variable(torch.tensor(one_hot_encoding(after_state), dtype=torch.float, device = device)).view((28*31,1))
    #print('actionCount:',actionCount)
    if(actionCount > 0):
        updateNeural(after_state,gamma, lam, alpha1, alpha2)
    #move = possible_moves[np.random.randint(len(possible_moves))]
    actionCount += 1

    return action

def gameFinishedUpdate(winner):
    global Z_w2,Z_w1,Z_b1,Z_b2,xold,w1,w2,b1,b2,epsilon
    reward = 1 if winner == 1 else 0
    #print('Game over reward:',reward)
    if actionCount > 1:
        gamma = 1
        h = torch.mm(w1,xold) + b1 # matrix-multiply x with input weight w1 and add bias
        h_sigmoid = h.sigmoid() # squash this with a sigmoid function
        y = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
        y_sigmoid = y.sigmoid() # squash the output
        delta2 = reward + gamma * 0 - y_sigmoid.detach().cpu().numpy()  # this is the usual TD error
        # using autograd and the contructed computational graph in pytorch compute all gradients
        y_sigmoid.backward()
        # update the eligibility traces
        Z_w2 = gamma * lam * Z_w2 + w2.grad.data
        Z_b2 = gamma * lam * Z_b2 + b2.grad.data
        Z_w1 = gamma * lam * Z_w1 + w1.grad.data
        Z_b1 = gamma * lam * Z_b1 + b1.grad.data
        # zero the gradients
        w2.grad.data.zero_()
        b2.grad.data.zero_()
        w1.grad.data.zero_()
        b1.grad.data.zero_()
        # perform now the update of weights
        delta2 =  torch.tensor(delta2, dtype = torch.float, device = device)
        w1.data = w1.data + alpha1 * delta2 * Z_w1
        b1.data = b1.data + alpha1 * delta2 * Z_b1
        w2.data = w2.data + alpha2 * delta2 * Z_w2
        b2.data = b2.data + alpha2 * delta2 * Z_b2

    #device = torch.device('cuda') 

    #import time
    #start = time.time()
    #training_steps = 40000
    #end = time.time()
    #print(end - start)
