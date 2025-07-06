#importing the module
import hex_engine as engine
import pandas as pd
#initializing a game object
game = engine.hexPosition()

from group_M.facade import Agent as agent_M

import pandas as pd


#Play a set of 500 games (250 times as black and 250 as white) 

def agents_play(white_player,black_player):
    white_wins = 0
    black_wins = 0
    for i in range(250):
        game.machine_vs_machine(machine1=white_player, machine2=black_player)
        if game.winner == 1:
            white_wins+= 1            
        if game.winner == -1: 
            black_wins+= 1
    return(white_wins, black_wins)

def game_set(agent1, agent2):
    
    # agent1 is white player/ agent2 is black player
    agent1_white_score, agent2_black_score = agents_play(agent1, agent2)

    # agent2 is white player/ agent1 is black player
    agent2_white_score, agent1_black_score = agents_play(agent2, agent1)

    agent1_score = agent1_white_score+ agent1_black_score
    agent2_score = agent2_white_score+ agent2_black_score

    if agent1_score!=agent2_score:  
        if agent1_score > agent2_score:
            print('SET Winner =', agent1)
            print('Final Score:', agent1_score)
            print(agent1_white_score, 'wins as WHITE PLAYER')
            print(agent1_black_score, 'wins as BLACK PLAYER')

        else:
            print('SET Winner =', agent2)
            print('Final Score:', agent2_score + 1)
            print(agent2_white_score, 'wins as WHITE PLAYER')
            print(agent2_black_score, 'wins as BLACK PLAYER')

    else:
        print('Draw: both agents won same amount of games')

    return(agent1_score, agent2_score)


agents_play(agent_M,None)
