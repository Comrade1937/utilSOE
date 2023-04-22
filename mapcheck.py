import cv2
import numpy as np 
import re 
import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(
                    description='Verification of SOE map',
                    epilog='Workers of the world unite !')
    parser.add_argument("map_png", type=str, help="path to provinces.png file")
    parser.add_argument("lua_file",type=str, help="path to provinces.lua file")
    parser.add_argument("-f","--fancy", 
            help=" on fail creates png, to be included in relese upon speedup",
            action="store_true")
    args = parser.parse_args(argv)

    img1 = cv2.imread(args.map_png)
    img = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)

    vertical_img = img.reshape((-1,3))
    vertical_24 = np.dot(vertical_img.astype(np.uint32),[1,256,65536])
    vunique24 = np.unique(vertical_24)

    C1 = np.bitwise_and(vunique24, 0xff)      
    C2 = np.bitwise_and(np.right_shift(vunique24,8),  0xff)   
    C3 = np.bitwise_and(np.right_shift(vunique24,16), 0xff)   
    unique_rgb = np.stack((C1, C2, C3),axis=-1)

    rgb2hex = lambda r,g,b: '%02x%02x%02x' %(r,g,b)
    urgb = [ rgb2hex(*i) for i in unique_rgb]

    with open(args.lua_file) as file: 
        read_content = file.read()
    result = re.findall(r"color=.{8}(?!,terrain=tt_sea)",read_content) 
    result = [re.sub("color=0x","",i) for i in result]

    result_sea = re.findall(r"color=.{8}(?=,terrain=tt_sea)",read_content) 
    result_sea = [re.sub("color=0x","",i) for i in result_sea]
    print("="*40)
    print("\u2605 Found in LUA \u2605".center(40," "))
    l_lua_prov = len(result)
    print("    provinces = ",str(l_lua_prov).rjust(5," "))
    l_lua_see = len(result_sea)
    print("sea provinces = ",str(l_lua_see).rjust(5," "))
    print("        total = ",str(l_lua_prov+l_lua_see).rjust(5," ") )

    sall = set(result)
    salls = set(result_sea)
    z = sall.intersection(salls)
    l_overlap = len(z)
    no_see_land_overlap = not bool(l_overlap)
    print(" same colour found in sea and land province ", 
            not no_see_land_overlap, end=" ")

    if no_see_land_overlap:
        print("\x1b[1;37;42m"+" PASSED! "+'\x1b[0m')
    else:
        print("\x1b[1;37;41m"+" FAILED! "+'\x1b[0m')
        print(f" overlap found in {l_overlap} instances")

    if not no_see_land_overlap:
        with open('overlap_color_codes.txt', 'w') as fff:
            print(list(z), file=fff)

    surgb = set(urgb)
    result2 = list()  
    result2_see = list()  
    for x in urgb:
        if x in result:
            result2.append(x)
        if x in result_sea:
            result2_see.append(x)

    print("="*40)
    print("\u2605 Found in PNG \u2605".center(40," "))
    nurgb = len(urgb)
    print("    unique rgb map colours = ", str(nurgb).rjust(5," "))  
    nprov = len(result2)
    print("    found in lua provinces = ", str(nprov).rjust(5," "))
    nsee = len(result2_see)
    print("found in lua see provinces = ", str(nsee).rjust(5," "))
    nmiss = nurgb-nprov-nsee
    print("                   MISSING = ", str(nmiss).rjust(5," "))

    no_map_lua_mismatch = not bool(nmiss)
    print("All map colours found in lua : ",no_map_lua_mismatch,end=" ")
    if no_map_lua_mismatch:
        print("\x1b[1;37;42m"+" PASSED! "+'\x1b[0m')
    else: 
        print("\x1b[1;37;41m"+" FAILED! "+'\x1b[0m')
    print("="*40)
    print(" MAP TEST STATUS : ",end=" ")
    check_status = no_map_lua_mismatch and no_see_land_overlap
    if check_status:
        print("\x1b[1;37;42m"+" PASSED! "+'\x1b[0m')
    else: 
        print("\x1b[1;37;41m"+" FAILED! "+'\x1b[0m') 
        print(" FIX THE MAP !".center(40," "))
    print("="*40)

    s1 = surgb.difference(sall) 
    s2 = s1.difference(salls)
    s2 = list(s2)

    if not no_map_lua_mismatch:
        with open('missing_color_codes.txt', 'w') as ffff:
            print(s2, file=ffff)

    if args.fancy and (not check_status):
        print(" Not implemented have patience ")


if __name__ == "__main__":
    main()


