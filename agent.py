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

theta = 0
alpha1 = 0.1
alpha2 = 0.1
epsilon = 0.1

#import time
#start = time.time()
#end = time.time()
#print(end - start)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# cuda will only create a significant speedup for large/deep networks and batched training
#device = torch.device('cuda') 



# define the parameters for the single hidden layer feed forward neural network
# randomly initialized weights with zeros for the biases
w1 = Variable(torch.randn(28*28,28*31, device = device, dtype=torch.float), requires_grad = True)
#b1 = Variable(torch.zeros((9*9,1), device = device, dtype=torch.float), requires_grad = True)
w2 = Variable(torch.randn(1, 28*28, device = device, dtype=torch.float), requires_grad = True)
#b2 = Variable(torch.zeros((1,1), device = device, dtype=torch.float), requires_grad = True)

# this function is used to prepare the raw board as input to the network
# for some games (not here) it may be useful to invert the board and see it from the perspective of "player"
def one_hot_encoding(board, player):
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


def epsilon_nn_greedy(board, possible_moves, possible_boards, player, epsilon):
    if (np.random.uniform() < epsilon):
        #print("explorative move")
        return possible_moves[np.random.randint(len(possible_moves))]

    va = np.zeros(len(possible_moves))
    
    for i in range(0,len(possible_moves)):
    
        x = Variable(torch.tensor(one_hot_encoding(possible_boards[i], player), dtype = torch.float, device = device)).view(28*31,1)
        
        h1 = torch.mm(w1,x)
        h1 = h1.sigmoid()
        
        h2 = torch.mm(w2,h1)
        
        y = h2.sigmoid()
        #print(y_pred)
        
        va[i] = y
        
    print(va)
    
    bestMove = np.argmax(va)
    
    if(possible_boards[bestMove][27] == 15):
        reward = 1.0
        #print(reward)
    
    if(possible_boards[bestMove][28] == -15):
        reward = -1.0
        #print(reward)
    
    return possible_moves[bestMove]


def action(board_copy,dice,player,i):
    # the champion to be
    # inputs are the board, the dice and which player is to move
    # outputs the chosen move accordingly to its policy
    
    
    # check out the legal moves available for the throw
    possible_moves, possible_boards = Backgammon.legal_moves(board_copy, dice, player)
    
    # if there are no moves available
    if len(possible_moves) == 0: 
        return [] 
    
    # make the best move according to the policy
    
    # policy missing, returns a random move for the time being
    #
    #
    #
    #
    #
    #print('_________')
    #Backgammon.pretty_print(board_copy)
    print(len(possible_boards))
    
    #print(len(possible_moves))
    #print(one_hot_encoding(board_copy, player))
    #print('_________')
    
    
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    #N, D_in, H, D_out = len(possible_boards), 31*28, 28*28, len(possible_boards)
    
   # x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
   # y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

   # w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
   # w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)
   

    action = epsilon_nn_greedy(board_copy, possible_moves, possible_boards, player, epsilon)
    
    
    
    #move = possible_moves[np.random.randint(len(possible_moves))]
    
    return action

