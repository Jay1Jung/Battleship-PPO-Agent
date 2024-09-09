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

#create_map(10)

#player_map = copy.deepcopy(map)
#player_record = copy.deepcopy(map)
#opp_map = copy.deepcopy(map)
#opp_record = copy.deepcopy(map)

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
    

    # 추가해야 하는게 헤드랑 테일이랑 정확하게 4칸 차이나야함1

  


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
                    if player_map[head_row-1][i] != "O" : # 이구문이 문젠데 지금
                        print("the point %d,%d had deployed before" %(head_row,i))
                        return set_location_battleship(player_map)
                    else:
                        player_map[head_row-1][i] = "B"

            if head_col > tail_col:
                for i in range(tail_col, head_col-1):
                    if player_map[head_row-1][i] != "O" : # 이구문이 
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

    head_row = input("Enter corrdinate for head row coordinate of the Submarine:")
    head_row = int(head_row)
    head_col = int(input("Enter corrdinate for head column coordinate of the Submarine:"))
    tail_row = int(input("Enter corrdinate for tail row coordinate of the Submarine:"))
    tail_col = int(input("Enter corrdinate for tail column coordinate of the Submarine:"))




    if head_row == tail_row and head_col == tail_col:
        raise ValueError
    
    if head_row != tail_row and head_col != tail_col:
        raise ValueError
    
    if abs(head_row - tail_row) != 1 and abs(head_col - tail_col) != 1:
        raise ValueError




    if 0 < head_row <= 10 and 0 < head_col <= 10 and 0 < tail_row <= 10 and 0 < tail_col <= 10: 
        player_map[head_row-1][head_col-1] = "C"
        player_map[tail_row-1][tail_col-1] = "C"

        if head_row == tail_row:
            if tail_col > head_col:
                for i in range(head_col,tail_col-3):
                    if player_map[head_row-1][i] != "O" : # 이구문이 
                        print("the point %d,%d had deployed before" %(head_row,i))
                        return set_location_Cruiser(player_map)
                    else:
                        player_map[head_row-1][i] = "C"

            if head_col > tail_col:
                for i in range(tail_col, head_col-3):
                    if player_map[head_row-1][i] != "O" : # 이구문이 
                        print("the point %d,%d had deployed before" %(head_row,i))
                        return set_location_Cruiser(player_map)
                    else:    
                        player_map[head_row-1][i] = "C"


        if head_col == tail_col:
            if tail_row > head_row:
                for i in range(head_row,tail_row-3):
                    if player_map[i][head_col-1] != "O" : # 이구문이 
                        print("the point %d,%d had deployed before" %(head_row,i))
                        return set_location_Cruiser(player_map)
                    else:
                        player_map[i][head_col-1] = "C"

            if head_row > tail_row:
                for i in range(tail_row, head_row-3):
                    if player_map[i][head_col-1] != "O" : # 이구문이 
                        print("the point %d,%d had deployed before" %(head_row,i))
                        return set_location_Cruiser(player_map)
                    
                    else:
                        player_map[i][head_col-1] = "C"

    return player_map



opp_address = []
def comp_set_location_aircraft(opp_map):
    # 일단 두개의 수를 랜덤으로 뽑아서 헤드값 좌표로 설정하구, 그다음에 1이나 2가 나왔을때, 로우를 고정할지 콜 을 고정할지 결정하는 걸루 하자.
    head_row = randrange(0,10)
    head_col = randrange(0,10)

    opp_map[head_row][head_col] = "A"



    dice = randrange(0,2)  # row 고정
    if dice == 0:
        tail_row = head_row
        if head_col + 4 < 9:
            tail_col = head_col + 4 
        if head_col - 4 > 0:
            tail_col = head_col - 4

        if head_col > tail_col:
            for i in range(tail_col, head_col+1):
                opp_map[head_row][i] = "A"
                opp_address.append([head_row, i])
        if tail_col > head_col:
            for i in range(head_col,tail_col+1):
                opp_map[head_row][i] = "A"
                opp_address.append([head_row,i])

    if dice == 1:
        tail_col = head_col

        if head_row + 4 < 9:
            tail_row = head_row + 4
        if head_row - 4 > 0:
            tail_row = head_row - 4
    
        if head_row > tail_row:
            for i in range(tail_row,head_row+1):
                opp_map[i][head_col] = "A"
                opp_address.append([i,head_col])
        if tail_row > head_row: 
            for i in range(head_row,tail_row+1):
                opp_map[i][head_col] = "A"
                opp_address.append([i,head_col])

    return opp_map
#comp_set_location(map)
#print_map(map)

def comp_set_location_battleship(opp_map):
    # 일단 두개의 수를 랜덤으로 뽑아서 헤드값 좌표로 설정하구, 그다음에 1이나 2가 나왔을때, 로우를 고정할지 콜 을 고정할지 결정하는 걸루 하자.

    while True:

        head_row = randrange(0,10)
        head_col = randrange(0,10)

        if [head_row,head_col] not in opp_address:
            break

    opp_map[head_row][head_col] = "B"
    opp_address.append[head_row,head_col]




    dice = randrange(0,2)  # row 고정
    if dice == 0:
        tail_row = head_row
        if head_col + 3 < 9:
            tail_col = head_col + 3 
        if head_col - 3 > 0:
            tail_col = head_col - 3

        if head_col > tail_col:
            for i in range(tail_col, head_col+1):
                if [head_row,i] not in opp_address:
                    opp_map[head_row][i] = "B"



                
        if tail_col > head_col:
            for i in range(head_col,tail_col+1):
                if opp_map[head_row][i] =="O":
                    opp_map[head_row][i] = "B"


    if dice == 1:
        tail_col = head_col

        if head_row + 3 < 9:
            tail_row = head_row + 3
        if head_row - 3 > 0:
            tail_row = head_row - 3
    
        if head_row > tail_row:
            for i in range(tail_row,head_row+1):
                if [i,head_col] not in opp_address:
                    opp_map[i][head_col] = "B"

        if tail_row > head_row: 
            for i in range(head_row,tail_row+1):
                if opp_map[i][head_col] =="O":
                    opp_map[i][head_col] = "B"
                else: 
                    return comp_set_location_battleship(opp_map)

    return opp_map


def comp_set_location_submarine(opp_map):
    # 일단 두개의 수를 랜덤으로 뽑아서 헤드값 좌표로 설정하구, 그다음에 1이나 2가 나왔을때, 로우를 고정할지 콜 을 고정할지 결정하는 걸루 하자.

    while True:
        head_row = randrange(0,10)
        head_col = randrange(0,10)

        if [head_row,head_col] not in opp_address:
            opp_map[head_row][head_col] = "S"

            break

    opp_address.append([head_row,head_col])



    dice = randrange(0,2)  # row 고정
    if dice == 0:
        tail_row = head_row
        if head_col + 2 < 9:
            tail_col = head_col + 2 
        if head_col - 2 > 0:
            tail_col = head_col - 2

        if head_col > tail_col:
            for i in range(tail_col, head_col+1):
                if [head_row,i] in opp_address:
                    return comp_set_location_submarine
                
                else: 
                    opp_map[head_row][i] = "S"

        
                
        if tail_col > head_col:
            for i in range(head_col,tail_col+1):
                if [head_row,i] in opp_address:
                    return comp_set_location_submarine
                
                else: 
                    opp_map[head_row][i] = "S"
                    opp_address.append([head_row,i])

    if dice == 1:
        tail_col = head_col

        if head_row + 2 < 9:
            tail_row = head_row + 2
        if head_row - 2 > 0:
            tail_row = head_row - 2
    
        if head_row > tail_row:
            for i in range(tail_row,head_row+1):
                if [i,head_col] in opp_address:
                    return comp_set_location_submarine(opp_map)
                else:
                    opp_map[i][head_col] = "S"
                    opp_address.append([i,head_col])
                
        if tail_row > head_row: 
            for i in range(head_row,tail_row+1):
                if [i,head_col] in opp_address:
                    return comp_set_location_submarine(opp_map)
                else:
                    opp_map[i][head_col] = "S"
                    opp_address.append([i,head_col])

    return opp_map


def comp_set_location_destroyer(opp_map):
    # 일단 두개의 수를 랜덤으로 뽑아서 헤드값 좌표로 설정하구, 그다음에 1이나 2가 나왔을때, 로우를 고정할지 콜 을 고정할지 결정하는 걸루 하자.
    head_row = randrange(0,10)
    head_col = randrange(0,10)

    if opp_map[head_row][head_col] == "O":
        opp_map[head_row][head_col] = "D"

    else: 
        return comp_set_location_destroyer(opp_map)



    dice = randrange(0,2)  # row 고정
    if dice == 0:
        tail_row = head_row
        if head_col + 2 < 9:
            tail_col = head_col + 2 
        if head_col - 2 > 0:
            tail_col = head_col - 2

        if head_col > tail_col:
            for i in range(tail_col, head_col+1):
                if opp_map[head_row][i] == "O":
                    opp_map[head_row][i] = "D"

                else: 
                    return comp_set_location_destroyer(opp_map)
                
        if tail_col > head_col:
            for i in range(head_col,tail_col+1):
                if opp_map[head_row][i] =="O":
                    opp_map[head_row][i] = "D"
                else:
                    return comp_set_location_destroyer(opp_map)

    if dice == 1:
        tail_col = head_col

        if head_row + 2 < 9:
            tail_row = head_row + 2
        if head_row - 2 > 0:
            tail_row = head_row - 2
    
        if head_row > tail_row:
            for i in range(tail_row,head_row+1):
                if opp_map[i][head_col] == "O":
                    opp_map[i][head_col] = "D"
                else:
                    return comp_set_location_destroyer(opp_map)
        if tail_row > head_row: 
            for i in range(head_row,tail_row+1):
                if opp_map[i][head_col] =="O":
                    opp_map[i][head_col] = "D"
                else: 
                    return comp_set_location_destroyer(opp_map)

    return opp_map

def comp_set_location_cruiser(opp_map):
    # 일단 두개의 수를 랜덤으로 뽑아서 헤드값 좌표로 설정하구, 그다음에 1이나 2가 나왔을때, 로우를 고정할지 콜 을 고정할지 결정하는 걸루 하자.
    head_row = randrange(0,10)
    head_col = randrange(0,10)

    if opp_map[head_row][head_col] == "O":
        opp_map[head_row][head_col] = "C"

    else: 
        return comp_set_location_cruiser(opp_map)



    dice = randrange(0,2)  # row 고정
    if dice == 0:
        tail_row = head_row
        if head_col + 1 < 9:
            tail_col = head_col + 1 
        if head_col - 1 > 0:
            tail_col = head_col - 1

        if head_col > tail_col:
            for i in range(tail_col, head_col+1):
                if opp_map[head_row][i] == "O":
                    opp_map[head_row][i] = "C"

                else: 
                    return comp_set_location_cruiser(opp_map)
                
        if tail_col > head_col:
            for i in range(head_col,tail_col+1):
                if opp_map[head_row][i] =="O":
                    opp_map[head_row][i] = "C"
                else:
                    return comp_set_location_cruiser(opp_map)

    if dice == 1:
        tail_col = head_col

        if head_row + 1 < 9:
            tail_row = head_row + 1
        if head_row - 1 > 0:
            tail_row = head_row - 1
    
        if head_row > tail_row:
            for i in range(tail_row,head_row+1):
                if opp_map[i][head_col] == "O":
                    opp_map[i][head_col] = "C"
                else:
                    return comp_set_location_cruiser(opp_map)
        if tail_row > head_row: 
            for i in range(head_row,tail_row+1):
                if opp_map[i][head_col] =="O":
                    opp_map[i][head_col] = "C"
                else: 
                    return comp_set_location_cruiser(opp_map)

    return opp_map

# 이 4개의 함수는 그냥 ㅈ됨 시발 첨 부터 다시 할것.

def check_player_hit(opp_map , player_record):
    #while True:
    row = int(input("Guess the row"))
    col = int(input("Guess the column"))

    hit = 0

        #if player_record[row][col] == "X" or "A":
            #print("You Have've already shot the point") 
            #continue
        # 왜 여기가 자꾸 오류가 나지 ㅅㅂ?

    if opp_map[row-1][col-1] == "A" or "B" or "S" or "D" or "C":
        player_record[row-1][col-1] = "H"
        opp_map[row-1][col-1] == "H"
        hit += 1
        print("Opponent ship been hit")

    
            
            
    else: 
        player_record[row-1][col-1] = "M"
        print("Missed")

    return hit
    # 맞췄을때 기록에다가 맞췄다 라는걸 보여줄 필요가있음.
        
#check_player_hit(opp_map,player_record)
memory = []
def check_opp_hit(player_map, opp_record):
    #while True:
    row = randrange(0,10)
    col = randrange(0,10)
    hit = 0



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

#def target_hunt():
    """
    어떤 특정한 배를 타겟했을 경우에 맞았다 그러면 그 주변 4칸을 우선적으로 탐색하는 알고리즘

    근데 이 알고리즘을 실행시킬때 이미 위에 if 문으로 탐색된 점인지 안된점인지 확인하고 돌릴거임.
    그러니까 조건에 상관쓸 필요없이 하면 됨. 근데 문제는 어떻게 해야할까 흠....
    """
#  if check_opp_hit.


    




# 지금 넣어야 되는 기능이 모든 알파벳이 x가 되면 승리 선언 하는것 그리고 이 코드를 어떻게 돌린건지 한번 알아봐야함.



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


        if player_hit == 5:
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
        if opp_hit ==5:
            print("Opp win")
            break

        #if check_opp_hit is True:
        

        print("oppoenent's map")
        print_map(opp_map)
        print("oppoent's record")
        print_map(opp_record)
 
        
        count += 1
        print(count)


    



        




    

# 오류사항은 게임전용맵이랑 그 기록용맵이랑 혼용되서 사용되고 있어서 a와 m이 같은 곳에 나온다는 점을 수정해야함.








    



