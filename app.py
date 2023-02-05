import pandas as pd
import recommended_activity as re_ac
import streamlit as st
import os


@st.cache(allow_output_mutation=True)
#이미지 가져오기
def image_load(first_rec_list):
    images = []
    image_path = "./data/game_image/"
    for i in range(len(first_rec_list)):
        if os.path.exists(image_path + first_rec_list[i] + ".jpg"):
            images.append(image_path + first_rec_list[i] + ".jpg")
        else:
            images.append(image_path + "No_Image_Available.jpg")
    return images
    
# 초기 화면 설정 시작
st.set_page_config(layout="wide")
st.header('원하는 게임을 골라 주세요.')

recommended_game_list = re_ac.first_page_rec
images = image_load(recommended_game_list)
# -- 초기 화면 설정 끝


# 클릭했던 목록
click_list = []
idx = 0

for i in range(0, 2):
    cols = st.columns(5)
    for col in cols:
        col.image(images[idx])
        col.write(recommended_game_list[idx])

        choice = col.button('선택', key= idx) # 클릭 여부
        
        if choice:
            click_list.append(recommended_game_list[idx])
            recommend_games, top_user = re_ac.game_click_event(click_list)
            images = image_load(recommend_games)
            recommended_game_list = recommend_games

            for key in st.session_state.keys():
                del st.session_state[key]

            message = "{0}번 유저와 가장 흡사합니다.".format(top_user)
            st.write(message)

        idx += 1






# def screen_rec(game_list, imgs):
#     idx = 0 
#     for i in range(0, 2):
#         cols = st.columns(5)
#         for col in cols:
#             col.image(imgs[idx])
#             col.write(game_list[idx])
#             choice = col.button('선택', key= idx)
            
#             if choice:
#                 with st.spinner('please wait...'):
#                     recommend_games = re_ac.game_click_event(game_list[idx])
#                     images = image_load(recommend_games)
                    
#                     screen_rec(game_list = recommend_games, imgs= images)
                    
#             idx += 1

# screen_rec(game_list = first_rec, imgs= images, idx=0)

































