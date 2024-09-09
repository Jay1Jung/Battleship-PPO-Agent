from random import randrange 
import copy

memory = []
def random_hit(player_map, opp_record,memory):

    #while True:
    row = randrange(0,10)
    col = randrange(0,10)
    hit = 0
    count = 0



    if [row-1,col-1] not in memory:
        if player_map[row-1][col-1] == "A" or "B" or "C" or "D" or "S":
            opp_record[row-1][col-1] = "H"
            player_map[row-1][col-1] == "H"
            hit += 1
            memory.append([row-1,col-1])
            print("Ally ship has been hit")
        else: 
            opp_record[row-1][col-1] = "M"
            print("Enemy Missed")
    else: 
        return random_hit(player_map, opp_record,memory)
            
    return hit

