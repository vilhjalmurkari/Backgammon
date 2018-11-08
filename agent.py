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


dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

#import time
#start = time.time()
#end = time.time()
#print(end - start)


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


def epsilon_nn_greedy(board, possible_moves, possible_boards, player, epsilon,training_data, w1, w2):
    if (np.random.uniform() < epsilon):
        #print("explorative move")
        return possible_moves[np.random.randint(len(possible_moves))]
    
    #inntak
    x = Variable(torch.FloatTensor(training_data).type(dtype), requires_grad=False)
    #úttak
    y = Variable(torch.randn(len(possible_boards), len(possible_boards)).type(dtype), requires_grad=False)
    
    learning_rate = 1e-6
    
    v = []
    
    # Forward pass: compute predicted y using operations on Variables; these
    # are exactly the same operations we used to compute the forward pass using
    # Tensors, but we do not need to keep references to intermediate values since
    # we are not implementing the backward pass by hand.
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    #print(y_pred)
  
    # Compute and print loss using operations on Variables.
    # Now loss is a Variable of shape (1,) and loss.data is a Tensor of shape
    # (1,); loss.data[0] is a scalar value holding the loss.
    loss = (y_pred - y).pow(2).sum()
    #print(loss.data[0])
    
    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Variables with requires_grad=True.
    # After this call w1.grad and w2.grad will be Variables holding the gradient
    # of the loss with respect to w1 and w2 respectively.
    loss.backward()
    
    # Update weights using gradient descent; w1.data and w2.data are Tensors,
    # w1.grad and w2.grad are Variables and w1.grad.data and w2.grad.data are
    # Tensors.
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data
    
    # Manually zero the gradients 
    w1.grad.data.zero_()
    w2.grad.data.zero_()
    
    v = y_pred.data.numpy()
       
    final = v[len(v)-1]
    #print(final)
    return possible_moves[np.argmax(final)]

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
    #print(len(possible_boards))
    
    #print(len(possible_moves))
    #print(one_hot_encoding(board_copy, player))
    #print('_________')
    
    training_data = []
    for i in range(len(possible_moves)):
        one_hot = one_hot_encoding(possible_boards[i], player)
        training_data.append(one_hot)
    
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    #N, D_in, H, D_out = len(possible_boards), 31*28, 28*28, len(possible_boards)
    
   # x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
   # y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

   # w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
   # w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)
   
    w1 = Variable(torch.randn(28*31, 28*28).type(dtype), requires_grad=True)
    w2 = Variable(torch.randn(28*28, len(possible_boards)).type(dtype), requires_grad=True)
    
    
    action = epsilon_nn_greedy(board_copy, possible_moves, possible_boards, player, epsilon, training_data, w1, w2)
    
    
    
    
    #move = possible_moves[np.random.randint(len(possible_moves))]

    return action