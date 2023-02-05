import pandas as pd

rec_games_list = pd.read_pickle("./rec_game_list.pkl")

rec_games_list.loc[0]['Title']
rec_games_list.loc[1]['url']

import urllib.request
from bs4 import BeautifulSoup
import time

rec_games_list
error_image_name = []
for i in range(rec_games_list.shape[0]):
    
    url = rec_games_list.loc[i]['url']
    req = urllib.request.Request(url)
    res = urllib.request.urlopen(url).read()
    
    soup = BeautifulSoup(res,'html.parser')
    soup = soup.find("div",class_="glance_ctn")
    #img의 경로를 받아온다
    try:
        imgUrl = soup.find("img")["src"]
    except:
        continue
    
    
    name = rec_games_list.loc[i]['Title']
    #urlretrieve
    try:
        urllib.request.urlretrieve(imgUrl,  name +'.jpg')
    except:
        error_image_name.append(i)
    
    print("수행한 횟수 : {num}, 이름 : {game}".format(num= i, game = name) )
    time.sleep(5)    
    


error_image_name
#이미지 이름 변경목록    
rec_games_list.loc[335]['Title'] #Resident Evil Revelations 2
rec_games_list.loc[335]['url']
rec_games_list.loc[374]['Title'] #Resident Evil 6
rec_games_list.loc[374]['url']
rec_games_list.loc[476]['Title'] #Holy Potatoes A Weapon Shop
rec_games_list.loc[476]['url']
rec_games_list.loc[573]['Title'] #resident evil 4
rec_games_list.loc[573]['url']
rec_games_list.loc[576]['Title'] #Resident Evil Revelations
rec_games_list.loc[576]['url']
rec_games_list.loc[616]['Title'] #Resident Evil HD REMASTER
rec_games_list.loc[616]['url']
rec_games_list.loc[701]['Title'] #how do you Do It
rec_games_list.loc[701]['url']
rec_games_list.loc[822]['Title'] #Dwarfs
rec_games_list.loc[822]['url']
rec_games_list.loc[1137]['Title'] #eden
rec_games_list.loc[1137]['url']
rec_games_list.loc[1291]['Title'] #Retro Grade
rec_games_list.loc[1291]['url']



















