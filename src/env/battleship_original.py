map_size = 10
map = []
from random import randrange
import copy

def create_map(map_size:int) -> list:       
    """
    Function that creates the battleship map for user
    """
    for i in range(map_size):
        map.append(["O"] * map_size)

    return map

def print_map(map):
    for i in map:
        print(" ".join(i)) # 여기서 join은 리스트 안의 것들을 문자열로 합쳐주는 함수고, "" 는 어떤걸로 구분을 할지 알려주는 거임 지금은 공백이 그역할을 하는것임

#print_map(map)

def occupied(xcoord: int, ycoord: int ,map: list): 
    occ = 0
    if map[xcoord][ycoord] != "0":
        occ = 1
        return occ
    else: 
        return occ



def set_location_aircraft(player_map):
    # set location of aircraft carrier
    head_row = input("Enter corrdinate for head row coordinate of the Aircraft Carrier:")
    head_row = int(head_row)
    head_col = int(input("Enter corrdinate for head column coordinate of the Aircraft Carrier:"))
    tail_row = int(input("Enter corrdinate for tail row coordinate of the Aircraft Carrier:"))
    tail_col = int(input("Enter corrdinate for tail column coordinate of the Aircraft Carrier:"))

    #print(head_row,head_col,tail_row,tail_col)
    #print(player_map)




    if head_row == tail_row and head_col == tail_col:
        raise ValueError
    
    if head_row != tail_row and head_col != tail_col:
        raise ValueError
    
    if abs(head_row - tail_row) != 4 and abs(head_col - tail_col) != 4:
        raise ValueError




    if 0 < head_row <= 10 and 0 < head_col <= 10 and 0 < tail_row <= 10 and 0 < tail_col <= 10: 
        player_map[head_row-1][head_col-1] = "A"
        player_map[tail_row-1][tail_col-1] = "A"

        if head_row == tail_row:
            if tail_col > head_col:
                for i in range(head_col ,tail_col-1):
                    player_map[head_row-1][i] = "A"
            if head_col > tail_col:
                for i in range(tail_col , head_col-1):
                    player_map[head_row-1][i] = "A"

        if head_col == tail_col:
            if tail_row > head_row:
                for i in range(head_row ,tail_row-1):
                    player_map[i][head_col-1] = "A"
            if head_row > tail_row:
                for i in range(tail_row, head_row-1):
                    player_map[i][head_col-1] = "A"

    return player_map

def set_location_battleship(player_map):

    head_row = input("Enter corrdinate for head row coordinate of the Battleship:")
    head_row = int(head_row)
    head_col = int(input("Enter corrdinate for head column coordinate of the Battleship:"))
    tail_row = int(input("Enter corrdinate for tail row coordinate of the Battleship:"))
    tail_col = int(input("Enter corrdinate for tail column coordinate of the Battleship:"))




    if head_row == tail_row and head_col == tail_col:
        raise ValueError
    
    if head_row != tail_row and head_col != tail_col:
        raise ValueError
    
    if abs(head_row - tail_row) != 3 and abs(head_col - tail_col) != 3:
        raise ValueError




    if 0 < head_row <= 10 and 0 < head_col <= 10 and 0 < tail_row <= 10 and 0 < tail_col <= 10: 
        player_map[head_row-1][head_col-1] = "B"
        player_map[tail_row-1][tail_col-1] = "B"

        if head_row == tail_row:
            if tail_col > head_col:
                for i in range(head_col,tail_col-1):
                    if player_map[head_row-1][i] != "O" : 
                        print("the point %d,%d had deployed before" %(head_row,i))
                        return set_location_battleship(player_map)
                    else:
                        player_map[head_row-1][i] = "B"

            if head_col > tail_col:
                for i in range(tail_col, head_col-1):
                    if player_map[head_row-1][i] != "O" :  
                        print("the point %d,%d had deployed before" %(head_row,i))
                        return set_location_battleship(player_map)
                    else:    
                        player_map[head_row-1][i] = "B"


        if head_col == tail_col:
            if tail_row > head_row:
                for i in range(head_row,tail_row-1):
                    if player_map[i][head_col-1] != "O" : # 이구문이 
                        print("the point %d,%d had deployed before" %(head_row,i))
                        return set_location_battleship(player_map)
                    else:
                        player_map[i][head_col-1] = "B"

            if head_row > tail_row:
                for i in range(tail_row, head_row-1):
                    if player_map[i][head_col-1] != "O" : # 이구문이 
                        print("the point %d,%d had deployed before" %(head_row,i))
                        return set_location_battleship(player_map)
                    
                    else:
                        player_map[i][head_col-1] = "B"

    return player_map

#set_location(map)
#print_map(map)

def set_location_destroyer(player_map):

    head_row = input("Enter corrdinate for head row coordinate of the Destroyer:")
    head_row = int(head_row)
    head_col = int(input("Enter corrdinate for head column coordinate of the Destroyer:"))
    tail_row = int(input("Enter corrdinate for tail row coordinate of the Destroyer:"))
    tail_col = int(input("Enter corrdinate for tail column coordinate of the Destroyer:"))




    if head_row == tail_row and head_col == tail_col:
        raise ValueError
    
    if head_row != tail_row and head_col != tail_col:
        raise ValueError
    
    if abs(head_row - tail_row) != 2 and abs(head_col - tail_col) != 2:
        raise ValueError




    if 0 < head_row <= 10 and 0 < head_col <= 10 and 0 < tail_row <= 10 and 0 < tail_col <= 10: 
        player_map[head_row-1][head_col-1] = "D"
        player_map[tail_row-1][tail_col-1] = "D"

        if head_row == tail_row:
            if tail_col > head_col:
                for i in range(head_col,tail_col-1):
                    if player_map[head_row-1][i] != "O"  : # 이구문이 
                        print("the point %d,%d had deployed before" %(head_row,i))
                        return set_location_destroyer(player_map)
                    else:
                        player_map[head_row-1][i] = "D"

            if head_col > tail_col:
                for i in range(tail_col, head_col-1):
                    if player_map[head_row-1][i] != "O" : # 이구문이 
                        print("the point %d,%d had deployed before" %(head_row,i))
                        return set_location_destroyer(player_map)
                    else:    
                        player_map[head_row-1][i] = "D"


        if head_col == tail_col:
            if tail_row > head_row:
                for i in range(head_row,tail_row-1):
                    if player_map[i][head_col-1] != "O" : # 이구문이 
                        print("the point %d,%d had deployed before" %(head_row,i))
                        return set_location_destroyer(player_map)
                    else:
                        player_map[i][head_col-1] = "D"

            if head_row > tail_row:
                for i in range(tail_row, head_row-1):
                    if player_map[i][head_col-1] != "O" : # 이구문이 
                        print("the point %d,%d had deployed before" %(head_row,i))
                        return set_location_destroyer(player_map)
                    
                    else:
                        player_map[i][head_col-1] = "D"

    return player_map

def set_location_submarine(player_map):

    head_row = input("Enter corrdinate for head row coordinate of the Submarine:")
    head_row = int(head_row)
    head_col = int(input("Enter corrdinate for head column coordinate of the Submarine:"))
    tail_row = int(input("Enter corrdinate for tail row coordinate of the Submarine:"))
    tail_col = int(input("Enter corrdinate for tail column coordinate of the Submarine:"))




    if head_row == tail_row and head_col == tail_col:
        raise ValueError
    
    if head_row != tail_row and head_col != tail_col:
        raise ValueError
    
    if abs(head_row - tail_row) != 2 and abs(head_col - tail_col) != 2:
        raise ValueError




    if 0 < head_row <= 10 and 0 < head_col <= 10 and 0 < tail_row <= 10 and 0 < tail_col <= 10: 
        player_map[head_row-1][head_col-1] = "S"
        player_map[tail_row-1][tail_col-1] = "S"

        if head_row == tail_row:
            if tail_col > head_col:
                for i in range(head_col,tail_col-1):
                    if player_map[head_row-1][i] != "O" : # 이구문이 
                        print("the point %d,%d had deployed before" %(head_row,i))
                        return set_location_submarine(player_map)
                    else:
                        player_map[head_row-1][i] = "S"

            if head_col > tail_col:
                for i in range(tail_col, head_col-1):
                    if player_map[head_row-1][i] != "O" : # 이구문이 
                        print("the point %d,%d had deployed before" %(head_row,i))
                        return set_location_submarine(player_map)
                    else:    
                        player_map[head_row-1][i] = "S"


        if head_col == tail_col:
            if tail_row > head_row:
                for i in range(head_row,tail_row-1):
                    if player_map[i][head_col-1] != "O" : # 이구문이 
                        print("the point %d,%d had deployed before" %(head_row,i))
                        return set_location_submarine(player_map)
                    else:
                        player_map[i][head_col-1] = "S"

            if head_row > tail_row:
                for i in range(tail_row, head_row-1):
                    if player_map[i][head_col-1] != "O" : # 이구문이 
                        print("the point %d,%d had deployed before" %(head_row,i))
                        return set_location_submarine(player_map)
                    
                    else:
                        player_map[i][head_col-1] = "S"

    return player_map


def set_location_Cruiser(player_map):

    head_row = input("Enter corrdinate for head row coordinate of the Cruiser:")
    head_row = int(head_row)
    head_col = int(input("Enter corrdinate for head column coordinate of the Cruiser:"))
    tail_row = int(input("Enter corrdinate for tail row coordinate of the Cruiser:"))
    tail_col = int(input("Enter corrdinate for tail column coordinate of the Cruiser:"))


    if head_row == tail_row and head_col == tail_col:
        raise ValueError
    
    if head_row != tail_row and head_col != tail_col:
        raise ValueError
    
    if abs(head_row - tail_row) != 1 and abs(head_col - tail_col) != 1:
        raise ValueError




    if 0 < head_row <= 10 and 0 < head_col <= 10 and 0 < tail_row <= 10 and 0 < tail_col <= 10: 
        #player_map[head_row-1][head_col-1] = "C"
        #player_map[tail_row-1][tail_col-1] = "C"

        if head_row == tail_row:
            if tail_col > head_col:
                for i in range(head_col,tail_col+1):
                    if player_map[head_row-1][i] != "O" : # 이구문이 
                        print("the point %d,%d had deployed before" %(head_row,i))
                        return set_location_Cruiser(player_map)
                    else:
                        player_map[head_row-1][i] = "C"

            if head_col > tail_col:
                for i in range(tail_col, head_col+1):
                    if player_map[head_row-1][i] != "O" : # 이구문이 
                        print("the point %d,%d had deployed before" %(head_row,i))
                        return set_location_Cruiser(player_map)
                    else:    
                        player_map[head_row-1][i] = "C"


        if head_col == tail_col:
            if tail_row > head_row:
                for i in range(head_row,tail_row+1):
                    if player_map[i-1][head_col-1] != "O" : # 이구문이 
                        print("the point %d,%d had deployed before" %(head_row,i))
                        return set_location_Cruiser(player_map)
                    else:
                        player_map[i-1][head_col-1] = "C"

            if head_row > tail_row:
                for i in range(tail_row, head_row+1):
                    if player_map[i-1][head_col-1] != "O" : # 이구문이 
                        print("the point %d,%d had deployed before" %(head_row,i))
                        return set_location_Cruiser(player_map)
                    
                    else:
                        player_map[i-1][head_col-1] = "C"

    return player_map



opp_address = []
def comp_set_location_aircraft(opp_map):
    N = len(opp_map)
    tries = 0
    while True: 
        tries += 1
        head_row = randrange(0,N)
        head_col = randrange(0,N)

        dice = randrange(0,2)  # row 고정
        if dice == 0:
            tail_row = head_row
            if head_col + 4 < 9:
                tail_col = head_col + 4 
            elif head_col - 4 > 0:
                tail_col = head_col - 4
            else:
                continue

            coords = [(head_row, col) for col in range(min(head_col, tail_col), max(head_col, tail_col) + 1)]

        else:
            tail_col = head_col

            if head_row + 4 < 9:
                tail_row = head_row + 4
            elif head_row - 4 > 0:
                tail_row = head_row - 4
            else:
                continue

            coords = [(row, head_col) for row in range(min(head_row, tail_row), max(head_row, tail_row) + 1)]

        if any(opp_map[row][col] != "O" for row, col in coords):
                continue

        for r, c in coords:
            opp_map[r][c] = "A"
            opp_address.append([r, c])

        return opp_map




    return opp_map
#comp_set_location(map)
#print_map(map)

def comp_set_location_battleship(opp_map):

    while True:

        head_row = randrange(0,10)
        head_col = randrange(0,10)


        dice = randrange(0,2)  # row 고정
        if dice == 0:
            tail_row = head_row
            if head_col + 3 < 9:
                tail_col = head_col + 3 
            elif head_col - 3 > 0:
                tail_col = head_col - 3
            else:
                continue


            coords = [(head_row, col) for col in range(min(head_col, tail_col), max(head_col, tail_col) + 1)]


        else:
            tail_col = head_col

            if head_row + 3 < 9:
                tail_row = head_row + 3
            elif head_row - 3 > 0:
                tail_row = head_row - 3
            else :
                continue
        
            coords = [(row, head_col) for row in range(min(head_row, tail_row), max(head_row, tail_row) + 1)]

        
        if any(opp_map[row][col] != "O" for row, col in coords):
                continue

        for r, c in coords:
            opp_map[r][c] = "B"
            opp_address.append([r, c])

        return opp_map


def comp_set_location_submarine(opp_map):

    while True:
        head_row = randrange(0,10)
        head_col = randrange(0,10)

        dice = randrange(0,2)  # row 고정
        if dice == 0:
            tail_row = head_row
            if head_col + 2 < 9:
                tail_col = head_col + 2 
            elif head_col - 2 > 0:
                tail_col = head_col - 2
            else:
                continue
        
            coords = [(head_row, col) for col in range(min(head_col, tail_col), max(head_col, tail_col) + 1)]

        else:
            tail_col = head_col

            if head_row + 2 < 9:
                tail_row = head_row + 2
            elif head_row - 2 > 0:
                tail_row = head_row - 2
            else:
                continue

            coords = [(row, head_col) for row in range(min(head_row, tail_row), max(head_row, tail_row) + 1)]

        if any(opp_map[row][col] != "O" for row, col in coords):
            continue

        for r, c in coords:
            opp_map[r][c] = "S"
            opp_address.append([r, c])
        

        return opp_map


def comp_set_location_destroyer(opp_map):
    while True:
        head_row = randrange(0,10)
        head_col = randrange(0,10)

        dice = randrange(0,2)  # row 고정

        if dice == 0:
            tail_row = head_row
            if head_col + 2 < 9:
                tail_col = head_col + 2 
            elif head_col - 2 > 0:
                tail_col = head_col - 2
            else: 
                continue
            coords = [(head_row, col) for col in range(min(head_col, tail_col), max(head_col, tail_col) + 1)]

        else: 
            tail_col = head_col

            if head_row + 2 < 9:
                tail_row = head_row + 2
            elif head_row - 2 > 0:
                tail_row = head_row - 2
            else:
                continue

            coords = [(row, head_col) for row in range(min(head_row, tail_row), max(head_row, tail_row) + 1)]

        if any(opp_map[row][col] != "O" for row, col in coords):
            continue

        for r, c in coords:
            opp_map[r][c] = "D"
            opp_address.append([r, c])
        return opp_map
    
def comp_set_location_cruiser(opp_map):
    while True:
        head_row = randrange(0,10)
        head_col = randrange(0,10)

        dice = randrange(0,2)  # row 고정

        if dice == 0:
            tail_row = head_row
            if head_col + 1 < 9:
                tail_col = head_col + 1 
            elif head_col - 1 > 0:
                tail_col = head_col - 1
            else:
                continue

            coords = [(head_row, col) for col in range(min(head_col, tail_col), max(head_col, tail_col) + 1)]

        else:
            tail_col = head_col

            if head_row + 1 < 9:
                tail_row = head_row + 1
            elif head_row - 1 > 0:
                tail_row = head_row - 1
            else: 
                continue

            coords = [(row, head_col) for row in range(min(head_row, tail_row), max(head_row, tail_row) + 1)]


        if any(opp_map[row][col] != "O" for row, col in coords):
            continue

        for r, c in coords:
            opp_map[r][c] = "C"
            opp_address.append([r, c])

        return opp_map

def check_player_hit(opp_map , player_record):
    #while True:
    row = int(input("Guess the row"))
    col = int(input("Guess the column"))

    hit = 0

    if opp_map[row-1][col-1] in ["A", "B","S","D","C"]:
        player_record[row-1][col-1] = "H"
        opp_map[row-1][col-1] = "H"
        hit += 1
        print("Opponent ship been hit")

    
    else: 
        player_record[row-1][col-1] = "M"
        print("Missed")

    return hit
        
#check_player_hit(opp_map,player_record)
memory = []
def check_opp_hit(player_map, opp_record):
    #while True:
    row = randrange(0,10)
    col = randrange(0,10)
    hit = 0



    if [row-1,col-1] not in memory:
        if player_map[row-1][col-1] in ["A","B","D","S","C"]:
            opp_record[row-1][col-1] = "H"
            player_map[row-1][col-1] = "H"
            hit += 1
            memory.append([row-1,col-1])
            print("Ally ship has been hit")
        else: 
            opp_record[row-1][col-1] = "M"
            print("Enemy Missed")
    else: 
        return check_opp_hit(player_map, opp_record)
            
    return hit
    
#check_opp_hit(player_map, opp_record)
#print_map(opp_record)
    
def opp_random_shoot():
    row = randrange(0,10)
    col = randrange(0,10)

    return row, col

def random_improved():
    row = randrange(0,10,2)
    col = randrange(0,10,2)

    return row, col


if __name__ == "__main__":
    map = create_map(map_size)
    player_map = copy.deepcopy(map)
    player_record = copy.deepcopy(map)
    opp_map = copy.deepcopy(map)
    opp_record = copy.deepcopy(map)

    print("Players turn")
    set_location_aircraft(player_map)
    print_map(player_map)
    set_location_battleship(player_map)
    print_map(player_map)
    set_location_destroyer(player_map)
    print_map(player_map)
    set_location_submarine(player_map)
    print_map(player_map)
    set_location_Cruiser(player_map) 
    print_map(player_map)
    print("Player has completed deployment")

    print("\n Opponent's turn:")
    comp_set_location_aircraft(opp_map)
    comp_set_location_battleship(opp_map)
    comp_set_location_destroyer(opp_map)
    comp_set_location_submarine(opp_map)
    comp_set_location_cruiser(opp_map)
    print("Computer has completed deployment")


    player_hit = 0
    #dummy = 0
    opp_hit = 0 
    count = 0
    while True:



        
        print("Player's turn")

        player_hit += check_player_hit(opp_map,player_record) 


        if player_hit == 15:
            print("player win")
            break



        # if player_hit > dummy:
        #     print("player hit=",player_hit,"b=",dummy)
        #     dummy += check_player_hit(opp_map,player_record)
        #     continue


         
        print("Player map")
        print_map(player_map)
        print("Player record")
        print_map(player_record)
        print("player_hit=", player_hit)

        
    
        print("Opponent's turn")
        opp_hit += check_opp_hit(player_map,opp_record)
        if opp_hit ==15:
            print("Opp win")
            break

        #if check_opp_hit is True:
        

        print("oppoenent's map")
        print_map(opp_map)
        print("oppoent's record")
        print_map(opp_record)
 
        
        count += 1
        print(count)


    









    



