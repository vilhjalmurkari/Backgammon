#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
an example of an intelligent agent who flips the board
"""
import numpy as np
import Backgammon
import torch
from torch.autograd import Variable
from torch import optim

wtest =[]

actionCount = 0
reward = 0

def flip_board(board_copy):
    #flips the game board and returns a new copy
    idx = np.array([0,24,23,22,21,20,19,18,17,16,15,14,13,
    12,11,10,9,8,7,6,5,4,3,2,1,26,25,28,27])
    flipped_board = -np.copy(board_copy[idx])
        
    return flipped_board

def flip_move(move):
    if len(move)!=0:
        for m in move:
            for m_i in range(2):
                m[m_i] = np.array([0,24,23,22,21,20,19,18,17,16,15,14,13,
                                12,11,10,9,8,7,6,5,4,3,2,1,26,25,28,27])[m[m_i]]        
    return move

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

def epsilon_nn_greedy(board, possible_moves, possible_boards, player,model):
    va = np.zeros(len(possible_moves))
    xa = np.zeros((len(possible_moves),868))
    for i in range(0,len(possible_moves)):
        xa[i,:] = one_hot_encoding(possible_boards[i])
    x = Variable(torch.tensor(xa.transpose(), dtype = torch.float, device = model.device))
    h = torch.mm(model.w1,x) + model.b1
    h_sigmoid = h.sigmoid()
    #pi = torch.mm(theta,h_sigmoid).softmax(1)
    #xtheta_mean = torch.sum(torch.mm(h_sigmoid,torch.diagflat(pi)),1)
    #xtheta_mean = torch.unsqueeze(xtheta_mean,1)
    #m = torch.multinominal(pi,1)
    y = torch.mm(model.w2,h_sigmoid)+ model.b2
    va = y.sigmoid().detach().cpu()
    bestMove = np.argmax(va)
    return possible_boards[bestMove],possible_moves[bestMove]
  

def action(board_copy,dice,player,i,model):
    global actionCount
    # starts by flipping the board so that the player always sees himself as player 1
    if player == -1: board_copy = flip_board(board_copy)
        
    # check out the legal moves available for the throw
    possible_moves, possible_boards = Backgammon.legal_moves(board_copy, dice, player=1)
    
    # if there are no moves available, return an empty move
    if len(possible_moves) == 0: 
        return [] 
    
    # Make the bestmove:
    after_state,action = epsilon_nn_greedy(board_copy, possible_moves, possible_boards, player,model)
    #model.xtheta = xtheta_mean
    if(actionCount > 0):
        model.updateNeural(after_state)

    actionCount += 1

    
    model.xold = Variable(torch.tensor(one_hot_encoding(after_state), dtype=torch.float, device = model.device)).view((28*31,1))

    # if the table was flipped the move has to be flipped as well
    if player == -1: move = flip_move(action)
    
    return move
