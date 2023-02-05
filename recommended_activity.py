import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import random

result_R = pd.read_pickle("./data/result_R.pkl")
#토빅별
n1_topic_ratings = pd.read_pickle("./data/n1_topic_ratings.pkl")
n2_topic_ratings = pd.read_pickle("./data/n2_topic_ratings.pkl")
n3_topic_ratings = pd.read_pickle("./data/n3_topic_ratings.pkl")
topics_ratings = pd.concat([n1_topic_ratings, n2_topic_ratings, n3_topic_ratings])

def first_rec(open_rec_list):
    first_rec_list = random.sample(open_rec_list.Title.tolist(), k=10)
    return first_rec_list

#첫 화면 추천 게임들
first_page_rec = first_rec(topics_ratings)

#특정게임 클릭 이벤트 핸들러
def game_click_event(click_games):
    if len(click_games) == 1:
            
        #클릭한 게임에 playtime 상위 5위중 playtime비율 확률로 선택
        top_users = result_R[click_games[0]].sort_values(ascending=False)[:5]
        elements_prob = top_users / top_users.values.sum()
        choice_user = np.random.choice(elements_prob.index, p = elements_prob.values)
        result_R.loc[choice_user]
        
        #choice된 playtime이 가장높은 top10 게임리스트
        recommend_games = result_R.loc[choice_user].sort_values(ascending=False)[:10].index
        recommend_games = recommend_games.tolist()
    
        #추천 목록에서 이미 클릭했던 item은 중복 제거 후 다시 추천
        if click_games[0] in recommend_games:
            drop_recommend_games = result_R.iloc[choice_user].sort_values(ascending=False).drop(click_games[0])[:10].index
            recommend_games = drop_recommend_games.tolist()
    else:
        #클릭한 게임에 playtime 상위 10위중 playtime비율 확률로 선택
        top_users = result_R[click_games[-1]].sort_values(ascending=False)[:10]
        top_users_info = result_R.loc[top_users.index]
        
        #top유저중 지금까지 선택된 items의 평균중 가장 큰 유저 선택
        choice_user = top_users_info[click_games].mean(axis='columns').sort_values(ascending=False).index[0]
        
        #choice된 playtime이 가장높은 top10 게임리스트
        recommend_games = result_R.iloc[choice_user].sort_values(ascending=False)[:10].index
        recommend_games = recommend_games.tolist()
        
        #추천 목록에서 이미 클릭했던 items는 중복 제거 후 다시 추천
        if set(click_games) & set(recommend_games):
            dupli_items = set(click_games) & set(recommend_games)
            drop_recommend_games = result_R.iloc[choice_user].sort_values(ascending=False).drop(list(dupli_items))[:10].index
            recommend_games = drop_recommend_games.tolist()
            
    return recommend_games, choice_user

# click_list = []
# click_ = 'Dota 2'
# click_list.append(click_)
# game_click_event(click_list)



# def game_click_event(click_game):
#     #클릭한 게임에 대해 플레이시간이 높은 유저6 선택
#     top6_users = result_R[click_game].sort_values(ascending=False)[:6]
#     top6_users = top6_users.index
#     top6_users = result_R.loc[top6_users]
#     #랜덤으로 유저선택
#     recommend_games = top6_users.iloc[random.randrange(0, 6)].sort_values(ascending=False)[:10].index
#     recommend_games = recommend_games.tolist()
#     return recommend_games

















