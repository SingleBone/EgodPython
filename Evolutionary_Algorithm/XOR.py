#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 18:41:54 2019

@author: egod
"""

import neat
import os

def eval_genomes(genomes,config):
    for genomes_id,genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome,config)
        for xi,xo in zip(xor_in,xor_out):
            output = net.activate(xi)
            genome.fitness -= (output[0]-xo[0])**2
            
if __name__ == '__main__':
    
    xor_in = [(0,0),(1,0),(0,1),(1,1)]
    xor_out = [(0,),(1,),(1,),(0,)]
    
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir,'config-feedforward')
    config = neat.Config(neat.DefaultGenome,
                         neat.DefaultReproduction,
                         neat.DefaultSpeciesSet,
                         neat.DefaultStagnation,
                         config_file,
                         )
    
    p = neat.Population(config)
    winner = p.run(eval_genomes,300)
    
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner,config)
    for xi,xo in zip(xor_in,xor_out):
        output = winner_net.activate(xi)
        print("input: {!r}, excepted output: {!r}, got: {!r}".format(xi,xo,output))


    