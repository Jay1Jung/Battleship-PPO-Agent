map_size = 10
map = []
from random import randrange
import copy
opp_address = []


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

    
def comp_set_location_aircraft(opp_map, opp_address):
    head_row = randrange(0, 10)
    head_col = randrange(0, 10)
    
    # 방향을 결정 (0: 가로, 1: 세로)
    dice = randrange(0, 2)

    if dice == 0:  # 가로로 배치
        if head_col + 4 < 10:  # 오른쪽으로 배치 가능 (0,1,2,3,4,5)
            tail_col = head_col + 4
        else:  # 왼쪽으로 배치 (6,7,8,9)
            tail_col = head_col - 4
    else:  # 세로로 배치
        if head_row + 4 < 10:  # 아래로 배치 가능
            tail_row = head_row + 4
        else:  # 위로 배치
            tail_row = head_row - 4

    if dice == 0:  # 가로 배치
        for i in range(min(head_col, tail_col), max(head_col, tail_col) + 1):
            if [head_row, i] in opp_address:
                return comp_set_location_aircraft(opp_map, opp_address)
            opp_map[head_row][i] = "A"
            opp_address.append([head_row, i])
    else:  # 세로 배치
        for i in range(min(head_row, tail_row), max(head_row, tail_row) + 1):
            if [i, head_col] in opp_address:
                return comp_set_location_aircraft(opp_map, opp_address)
            opp_map[i][head_col] = "A"
            opp_address.append([i, head_col])

    return opp_map

def comp_set_location_battleship(opp_map, opp_address):
    head_row = randrange(0, 10)
    head_col = randrange(0, 10)
    
    # 방향을 결정 (0: 가로, 1: 세로)
    dice= randrange(0, 2)

    if dice == 0:  # 가로로 배치
        if head_col + 3 < 10:  # 오른쪽으로 배치 가능
            tail_col = head_col + 3
        elif head_col - 3 >= 0:  # 왼쪽으로 배치 가능
            tail_col = head_col - 3
        else:
            return comp_set_location_battleship(opp_map, opp_address)  # 유효하지 않으면 다시 시도
        tail_row = head_row  # 가로 배치이므로 행 고정

    else:  # 세로로 배치
        if head_row + 3 < 10:  # 아래로 배치 가능
            tail_row = head_row + 3
        elif head_row - 3 >= 0:  # 위로 배치 가능
            tail_row = head_row - 3
        else:
            return comp_set_location_battleship(opp_map, opp_address)  # 유효하지 않으면 다시 시도
        tail_col = head_col  # 세로 배치이므로 열 고정

    if dice == 0:  # 가로 배치
        for i in range(min(head_col, tail_col), max(head_col, tail_col) + 1):
            if [head_row, i] in opp_address:
                return comp_set_location_battleship(opp_map, opp_address)  # 충돌 시 다시 시도
            opp_map[head_row][i] = "B"
            opp_address.append([head_row, i])
    else:  # 세로 배치
        for i in range(min(head_row, tail_row), max(head_row, tail_row) + 1):
            if [i, head_col] in opp_address:
                return comp_set_location_battleship(opp_map, opp_address)  # 충돌 시 다시 시도
            opp_map[i][head_col] = "B"
            opp_address.append([i, head_col])
    return opp_map



def comp_set_location_submarine(opp_map, opp_address):
    head_row = randrange(0, 10)
    head_col = randrange(0, 10)
    
    # 방향을 결정 (0: 가로, 1: 세로)
    dice= randrange(0, 2)

    if dice == 0:  # 가로로 배치
        if head_col + 2 < 10:  # 오른쪽으로 배치 가능
            tail_col = head_col + 2
        elif head_col - 2 >= 0:  # 왼쪽으로 배치 가능
            tail_col = head_col - 2
        else:
            return comp_set_location_submarine(opp_map, opp_address)  # 유효하지 않으면 다시 시도
        tail_row = head_row  # 가로 배치이므로 행 고정

    else:  # 세로로 배치
        if head_row + 2 < 10:  # 아래로 배치 가능
            tail_row = head_row + 2
        elif head_row - 2 >= 0:  # 위로 배치 가능
            tail_row = head_row - 2
        else:
            return comp_set_location_submarine(opp_map, opp_address)  # 유효하지 않으면 다시 시도
        tail_col = head_col  # 세로 배치이므로 열 고정

    if dice == 0:  # 가로 배치
        for i in range(min(head_col, tail_col), max(head_col, tail_col) + 1):
            if [head_row, i] in opp_address:
                return comp_set_location_submarine(opp_map, opp_address)  # 충돌 시 다시 시도
            opp_map[head_row][i] = "S"
            opp_address.append([head_row, i])
    else:  # 세로 배치
        for i in range(min(head_row, tail_row), max(head_row, tail_row) + 1):
            if [i, head_col] in opp_address:
                return comp_set_location_submarine(opp_map, opp_address)  # 충돌 시 다시 시도
            opp_map[i][head_col] = "S"
            opp_address.append([i, head_col])
    return opp_map


def comp_set_location_destroyer(opp_map, opp_address):
    head_row = randrange(0, 10)
    head_col = randrange(0, 10)
    
    # 방향을 결정 (0: 가로, 1: 세로)
    dice= randrange(0, 2)

    if dice == 0:  # 가로로 배치
        if head_col + 2 < 10:  # 오른쪽으로 배치 가능
            tail_col = head_col + 2
        elif head_col - 2 >= 0:  # 왼쪽으로 배치 가능
            tail_col = head_col - 2
        else:
            return comp_set_location_destroyer(opp_map, opp_address)  # 유효하지 않으면 다시 시도
        tail_row = head_row  # 가로 배치이므로 행 고정

    else:  # 세로로 배치
        if head_row + 2 < 10:  # 아래로 배치 가능
            tail_row = head_row + 2
        elif head_row - 2 >= 0:  # 위로 배치 가능
            tail_row = head_row - 2
        else:
            return comp_set_location_destroyer(opp_map, opp_address)  # 유효하지 않으면 다시 시도
        tail_col = head_col  # 세로 배치이므로 열 고정

    if dice == 0:  # 가로 배치
        for i in range(min(head_col, tail_col), max(head_col, tail_col) + 1):
            if [head_row, i] in opp_address:
                return comp_set_location_destroyer(opp_map, opp_address)  # 충돌 시 다시 시도
            opp_map[head_row][i] = "D"
            opp_address.append([head_row, i])
    else:  # 세로 배치
        for i in range(min(head_row, tail_row), max(head_row, tail_row) + 1):
            if [i, head_col] in opp_address:
                return comp_set_location_destroyer(opp_map, opp_address)  # 충돌 시 다시 시도
            opp_map[i][head_col] = "D"
            opp_address.append([i, head_col])
    return opp_map


def comp_set_location_cruiser(opp_map, opp_address):
    head_row = randrange(0, 10)
    head_col = randrange(0, 10)
    
    # 방향을 결정 (0: 가로, 1: 세로)
    dice= randrange(0, 2)

    if dice == 0:  # 가로로 배치
        if head_col + 1 < 10:  # 오른쪽으로 배치 가능
            tail_col = head_col + 1
        elif head_col - 1 >= 0:  # 왼쪽으로 배치 가능
            tail_col = head_col - 1
        else:
            return comp_set_location_cruiser(opp_map, opp_address)  # 유효하지 않으면 다시 시도
        tail_row = head_row  # 가로 배치이므로 행 고정

    else:  # 세로로 배치
        if head_row + 1 < 10:  # 아래로 배치 가능
            tail_row = head_row + 1
        elif head_row - 1 >= 0:  # 위로 배치 가능
            tail_row = head_row - 1
        else:
            return comp_set_location_cruiser(opp_map, opp_address)  # 유효하지 않으면 다시 시도
        tail_col = head_col  # 세로 배치이므로 열 고정

    if dice == 0:  # 가로 배치
        for i in range(min(head_col, tail_col), max(head_col, tail_col) + 1):
            if [head_row, i] in opp_address:
                return comp_set_location_cruiser(opp_map, opp_address)  # 충돌 시 다시 시도
            opp_map[head_row][i] = "C"
            opp_address.append([head_row, i])
    else:  # 세로 배치
        for i in range(min(head_row, tail_row), max(head_row, tail_row) + 1):
            if [i, head_col] in opp_address:
                return comp_set_location_cruiser(opp_map, opp_address)  # 충돌 시 다시 시도
            opp_map[i][head_col] = "C"
            opp_address.append([i, head_col])
    return opp_map





map = create_map(map_size)
opp_map = copy.deepcopy(map)
opp_record = copy.deepcopy(map)

comp_set_location_aircraft(opp_map,opp_address)
comp_set_location_battleship(opp_map,opp_address)
comp_set_location_submarine(opp_map,opp_address)
comp_set_location_destroyer(opp_map,opp_address)
comp_set_location_cruiser(opp_map,opp_address)

print_map(opp_map)