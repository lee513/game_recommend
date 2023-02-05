import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

###source by 
###https://github.com/RussoMarioDamiano/Collaborative-Topic-Modeling

#데이터 불러오기
games = pd.read_csv("./steam_games.csv")
games
games.columns
games.info()
games['url'][0]
games['name']
games['types'].value_counts()
games.iloc[0]['popular_tags']
games.iloc[0]['game_description']
games.iloc[0]['minimum_requirements']

#토픽모델링에 필요한 요소만
games = games.loc[:, ["name", "game_description"]]

#name, description의 Nan 칼럼 확인및 제거
len(games[games['game_description'].isna()])
games = games[~games['game_description'].isna()]
len(games[games['name'].isna()])
games = games[~games['name'].isna()]
len(games)

#description의 시작부분 필요없는 문자열 제거
games['game_description'][0]
games['game_description'] = games['game_description'].apply(lambda x: x.replace(" About This Game ", ""))
games['game_description'] = games['game_description'].apply(lambda x: x.replace(" About This Content ", ""))
games['game_description']
#description의 빈곳 제거
games[games.game_description == " "]
games[games['name'] =='iREC']
games[games['name'] =='Sanator: Scarlet Scarf']
games = games[games.game_description != " "]
games.head(5)

#rating테이블 가져오기
ratings = pd.read_csv("./steam-200k.csv", header=None)
ratings.columns = ["UserID", "Title", "Action", "Value", "purchase_play"]
ratings.columns

#여기서는 play한 숫자만
ratings = ratings[ratings.Action != "purchase"]

#불필요한 columns제거
ratings = ratings.loc[:,["UserID", "Title", "Value"]]
#play시간 살펴보기
pd.DataFrame(ratings["Value"].describe()).T
ratings["Value"].value_counts()
ratings['Value'].max()
ratings['Value'].min()
ratings.Value.sort_values()
ratings.shape



#games 테이블과 ratings테이블 중복 제거와 게임 이름 추출

titles_game = set(games.name.to_list())
ratings.Title.value_counts()
titles_ratings = set(ratings.Title.to_list())
len(titles_game)
len(titles_ratings)

#game, ratings 동시에있는 game title 추출 
intersection = titles_game.intersection(titles_ratings)
len(intersection)

#intersection으로 재구축
games = games[games.name.isin(intersection)]

games

#게임의 이름이 중복이지만 개발자가 다른것을 모두 삭제
print(games[games.name.duplicated()].shape[0])
intersection = intersection.difference(set(games[games.name.duplicated()].name.to_list()))
games = games[games.name.isin(intersection)]
print(games[games.name.duplicated()].shape[0])

#ratings테이블도 재구축
ratings = ratings[ratings.Title.isin(intersection)]


#ratings와 games의 같은 title숫자 인지 확인
try:
    assert len(set(ratings.Title.to_list())) == games.shape[0]
    print("Dataset game entries match.")
except AssertionError:
    raise Exception("Dataset game entries differ.")

##ratings와 games
ratings
games


#title기준으로 플레이 시간
agg = ratings.groupby('Title')['Value'].agg({'sum'}).reset_index()
agg.head(5)

#특정 게임이 압도적으로 플레이 시간이 높다는것을 확인 할수 있다.
import matplotlib.pyplot as plt

plt.plot(agg['Title'], agg['sum'])
plt.title('each games with playtime')
plt.xlabel('games')
plt.ylabel('playtimes')
plt.ylim(agg['sum'].min(), agg['sum'].max())
plt.show()

#가장 플레이 시간이 높은 게임들
agg.sort_values(by=['sum'], ascending=False)

#play 시간 정규화
pt = ratings.Value
max_hours = pt.max()
min_hours = pt.min()
pt_scaled = (pt - min_hours) / (max_hours - min_hours )
ratings.Value = pt_scaled
ratings.head(5)

#유저 평점 배열 만들기
R = pd.pivot_table(data=ratings, values = ["Value"], index=["UserID"], columns=["Title"])
# remove the level on top of game names called "Value"
R.columns = R.columns.droplevel()
# remove leftover columns name from pivot operation
R.columns.name = ""
# lastly, fill in the NaNs with 0's
R.fillna(0, inplace=True)

#희소행렬
r = R.values
sparsity = float(len(r.nonzero()[0]))
sparsity /= (r.shape[0] * r.shape[1])
sparsity *= 100
print(f"Matrix sparsity: {round(sparsity, 2)}%")
r.shape

#지금까지 저장
#  "r" save
#np.save("./r.npy",r)
# "r" load
r = np.load("./r.npy")

# "ratings" save
#ratings.to_pickle("./ratings.pkl")
# "ratings" load
ratings = pd.read_pickle("./ratings.pkl")





# train - test split
def train_test_split(ratings, percs = [0.8, 0.2]):
    
    validation = np.zeros(ratings.shape)
    train = ratings.copy()
    
    for user in np.arange(ratings.shape[0]):
        val_ratings = np.random.choice(ratings[user,:].nonzero()[0],
                                        size = round(len(ratings[user,:].nonzero()[0]) * percs[1]),
                                        replace=False
                                        )
        train[user, val_ratings] = 0
        validation[user, val_ratings] = ratings[user, val_ratings]
    
    return train, validation

train, val = train_test_split(r)
print(f"""Train-test split exectuted.
      Train: {round(len(train.flatten().nonzero()[0]) / len(r.flatten().nonzero()[0]) * 100, 2)}% 
      Test: {round(len(val.flatten().nonzero()[0]) / len(r.flatten().nonzero()[0]) * 100, 2)}% """)



#############topic modeling#######사용
import spacy
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore, TfidfModel
import multiprocessing

nlp = spacy.load("en_core_web_sm")


# lemmatize
games["lemmas"] = [[[token.lemma_ if token.lemma_ != "-PRON-" else token.text.lower() 
                     for token in sentence if token.pos_ in {"NOUN", "VERB", "ADJ", "ADV", "X"}]
                    for sentence in nlp(speech).sents] for speech in games.game_description]
#games.to_pickle("steam_games_preprocessed.pkl")
games = pd.read_pickle("steam_games_preprocessed.pkl")

#lemmas전처리
stop_words = ['’', '-', 'co' , '[', '/h2][/h2', ']', '+' , '★', '|']
stop_words

instances = [[lemma for lemmatized_sentence in lemmatized_speech for lemma in lemmatized_sentence]
             for lemmatized_speech in games.lemmas]
len(instances)

def stop_wordf(instances):
    instances_ = []
    for _, docs in enumerate(instances):
        re_docs = [word for word in docs if not word in stop_words]
        instances_.append(re_docs)
    return instances_
        
instances = stop_wordf(instances)

dictionary = Dictionary(instances)

dictionary.filter_extremes(no_below = 5, no_above = 0.7)
print(dictionary)

ldacorpus = [dictionary.doc2bow(text) for text in instances]
ldacorpus[0]

ldacorpus[1498]
tfidfmodel = TfidfModel(ldacorpus)
model_corpus = tfidfmodel[ldacorpus]

#coherence로 최적 topic 갯수 찾기
from gensim.models import CoherenceModel
import gensim
coherence_score=[]
for i in range(2,15):
    model = gensim.models.ldamodel.LdaModel(corpus=model_corpus, id2word=dictionary, num_topics=i)
    coherence_model = CoherenceModel(model, texts = instances, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model.get_coherence()
    print('n=',i,'\nCoherence Score: ', coherence_lda)
    coherence_score.append(coherence_lda)

import matplotlib.pyplot as plt
import numpy

k=[]
for i in range(2,15):
    k.append(i)

x=numpy.array(k)
y=numpy.array(coherence_score)
plt.title('Topic Coherence')
plt.plot(x,y)
plt.xlim(2,15)
plt.xlabel('Number Of Topic (2-15)')
plt.ylabel('Cohrence Score')
plt.show()

#LDA 토픽모델링
num_topics = 10
num_passes = 30
chunk_size = len(model_corpus) * num_passes/200

ldamodel = LdaMulticore(num_topics=num_topics,
                    corpus=model_corpus,
                    id2word=dictionary,
                    workers=multiprocessing.cpu_count()-1,
                    chunksize=chunk_size,
                    passes=num_passes,
                    alpha=0.1)

ldamodel.save('gensim_model.gensim')
ldamodel = LdaMulticore.load('gensim_model.gensim')
ldamodel

toto = ldamodel.print_topics(num_words=5)
for topic in toto:
    print(topic)
ldamodel.show_topics()


all_topics = ldamodel.get_document_topics(model_corpus, per_word_topics=True, minimum_probability=0.0)

corpus_topics = []
for doc_topics, word_topics, phi_values in all_topics:
    corpus_topics.append([topic[1] for topic in doc_topics])

corpus_topics[0]
corpus_topics = np.array(corpus_topics)
corpus_topics.shape

#num_topics = 10
words = [ldamodel.print_topics(num_topics=num_topics, num_words=5)[topic][1].split('\"')[1::2] for topic in range(num_topics)]
words_per_topic = dict(zip(range(num_topics), words))
words_per_topic

#살펴보기
bow = dictionary.doc2bow(["kill"])
ldamodel.get_document_topics(bow )

#협업 토픽 모델링 theta
theta = corpus_topics.copy().T
theta = pd.DataFrame(theta)
theta = theta.values
theta.shape

#협업 토픽 모델링

###source by 
###https://github.com/RussoMarioDamiano/Collaborative-Topic-Modeling
X_train = train.copy()
X_val = val.copy()

from sklearn.metrics import mean_squared_error

def mse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return mean_squared_error(prediction, ground_truth)

from tqdm import trange
import sys

class CTM():
    """
    Collaborative Topic Modeling Model as developed by Wang and Blei (2012).
    Leverages topic proportions obtained from LDA model to improve predictions
    and allow for out-of-matrix predictions.
    
    Parameters:
        - sigma2: expected variance of ratings 
                  (variance of the ratings Normal prior)
        - sigma2_P: expected variance of the elements of the
                    preference vector
        - sigma2_Q: expected variance of the elements of the
                    quality vector
    """
    def __init__(self, epochs=200, learning_rate=0.001, sigma2=10, sigma2_P=10, sigma2_Q=10):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.sigma2 = sigma2
        self.sigma2_P = sigma2_P
        self.sigma2_Q = sigma2_Q
    
    
    def fit(self, theta, X_train, X_val):
        """
        Fit a CTM model.
        
        Parameters:
            - theta: (K X I) matrix of topic proportions obtained via LDA.
            - X_train: (U X I) ratings matrix to train the model on.
            - X_test: (U X I) ratings matrix to validate the model on.
        """
        
        K = theta.shape[0]
        U, I = X_train.shape
        
        #initialize P and Q matrices.
        # P is initialized randomly
        self.P = np.random.randint(0, 10) * np.random.rand(K, U)
        # Q is initialized to be equal to theta
        self.Q = theta.copy()
        
        self.train_error = []
        self.val_error = []
        
        # obtain the pairs of (u, i) indices for which we observe a rating
        users, items = X_train.nonzero()
        
        
        # begin training
        for iteration in trange(self.epochs, file=sys.stdout, desc='CTM'):
            for u, i in zip(users, items):
                error = X_train[u, i] - np.dot(self.P[:, u].T, self.Q[:, i])

                # we are MAXIMIZING the likelihood via gradient ascent
                self.P[:, u] += self.learning_rate * (-self.P[:, u]/self.sigma2_P + (self.P[:, u] * error) * self.sigma2)
                self.Q[:, i] += self.learning_rate * (-(self.Q[:, i] - theta[:, i])/self.sigma2_Q + (self.Q[:, i] * error) * self.sigma2)

            self.train_error.append(mse(np.dot(self.P.T, self.Q), X_train))
            self.val_error.append(mse(np.dot(self.P.T, self.Q), X_val))
    
    
    
    def predict_ratings(self):
        """
        Returns the matrix of predicted ratings.
        """
        return np.dot(self.P.T, self.Q)
    
    
    
    def predict_out_of_matrix(self, topics):
        """
        Returns the (U X 1) vector of predicted ratings 
        for an unrated item, using the item's topic proportions.
        
        Parameters:
            - topics: (K X 1) array of topic proportions
                      for the unrated item.
        """
        return np.dot(self.P.T, topics)
    

ctm = CTM(epochs=200, sigma2_P=5, sigma2_Q=5, sigma2=1)
ctm.fit(theta, X_train, X_val)

print(f"""Training done. 
        Train error: {ctm.train_error[-1]}
        Validation error: {ctm.val_error[-1]}""")

fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(ctm.train_error, label = "train_error")
ax.plot(ctm.val_error, label = "val_error")
ax.set_title("Collaborative Topic Modeling")
ax.set_ylabel("MSE")
ax.set_xlabel("Epochs")
ax.legend();


#학습된 ctm 저장
ctm_predict = ctm.predict_ratings()
#np.save('trained_ctm.npy', ctm_predict)
ctm_predict = np.load('./trained_ctm.npy')
ctm.predict_ratings().shape

#학습저장
#np.save('trained_P.npy', ctm.P)
#np.save('trained_Q.npy', ctm.Q)

fig, ax = plt.subplots()

# Example data
labels = list(words_per_topic.values())
performance = ctm.P.sum(axis=1)

ax.barh([", ".join(l) for l in labels], performance, align='center')
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Popularity')
ax.set_title('How Popular is each genre')

plt.show()


#특정게임 topic 연관성 보기
gamename = "Dota 2"

# find the index number of halflife
for i, t in enumerate(R.columns):
    if t == gamename:
        idx = i
        break        
print(f"Topic relevances for {gamename}:")        
for i, mixture in enumerate(ctm.Q[:, idx]):
    print(f"\t{round(mixture * 100, 2)}: {words_per_topic[i]}")

#학습한 데이터로 최종 플레이시간 R테이블 만들기
result_R = pd.DataFrame(ctm_predict, index = R.index, columns = R.columns)
result_R = result_R.reset_index()
result_R.drop(['UserID'], axis=1, inplace=True)

# "result_R" save
#result_R.to_pickle("./data/result_R.pkl")
# "result_R" load
result_R = pd.read_pickle("./data/result_R.pkl")

#유저별 모든 게임 플레이시간 합과 평균
result_R.sum(axis='columns').sort_values(ascending=False)
result_R.mean(axis='columns').sort_values(ascending=False)

import matplotlib.pyplot as plt
import numpy as np

result_R.shape
# "corpus_topics" save
#np.save("./corpus_topics.npy", corpus_topics)
# "corpus_topics" load
corpus_topics = np.load("./corpus_topics.npy")


#게임별 해당 토픽 할당
each_topic = np.argmax(corpus_topics, axis=1)
games['n_topic'] = each_topic

#1번 토픽이 압도적으로 많은걸 알 수 있다.
games.groupby('n_topic')['name'].agg({'count'}).reset_index()
ratings['Title'].value_counts()

#ratings와 game_topic 합치기
games.rename(columns={'name':'Title'}, inplace=True)
n_topic_game = games[['Title', 'n_topic']]
n_topic_ratings = pd.merge(ratings, n_topic_game, how='left', on='Title')
n_topic_ratings['n_topic'].value_counts()


# 토픽기준으로 게임별 유저들이 플레이한 횟수

# 토픽1에서 유저들이 많이 접한 게임 상위 100개 뽑기
n1_topic_ratings = n_topic_ratings[n_topic_ratings['n_topic'] == 1]
n1_topic_ratings = n1_topic_ratings.groupby('Title')['UserID'].agg({'count'}).reset_index().sort_values(by = 'count', ascending=False)
n1_topic_ratings = n1_topic_ratings[:100]
print(n1_topic_ratings)
# "n1_topic_ratings" save
#n1_topic_ratings.to_pickle("./n1_topic_ratings.pkl")
# "n1_topic_ratings" load
n1_topic_ratings = pd.read_pickle("./n1_topic_ratings.pkl")

# 토픽4에서 유저들이 많이 접한 게임 상위 20개 뽑기
n2_topic_ratings = n_topic_ratings[n_topic_ratings['n_topic'] == 4]
n2_topic_ratings = n2_topic_ratings.groupby('Title')['UserID'].agg({'count'}).reset_index().sort_values(by = 'count', ascending=False)
n2_topic_ratings = n2_topic_ratings[:20]
print(n2_topic_ratings)
# "n2_topic_ratings" save
#n2_topic_ratings.to_pickle("./n2_topic_ratings.pkl")
# "n1_topic_ratings" load
n2_topic_ratings = pd.read_pickle("./n2_topic_ratings.pkl")

# 토픽7에서 유저들이 많이 접한 게임 상위2개 뽑기
n3_topic_ratings = n_topic_ratings[n_topic_ratings['n_topic'] == 7]
n3_topic_ratings = n3_topic_ratings.groupby('Title')['UserID'].agg({'count'}).reset_index().sort_values(by = 'count', ascending=False)
print(n3_topic_ratings)
# "n1_topic_ratings" save
#n3_topic_ratings.to_pickle("./n3_topic_ratings.pkl")
# "n1_topic_ratings" load
n3_topic_ratings = pd.read_pickle("./n3_topic_ratings.pkl")


#추천에 사용할 게임 목록들
no_dupli_ratings = ratings.drop_duplicates(subset=["Title"], keep="first")
rec_games_list = pd.merge(no_dupli_ratings, games, how='left', left_on='Title', right_on='name')
rec_game_list = rec_games_list[['Title', 'url']]
#rec_game_list.to_pickle("./rec_game_list.pkl")
rec_games = pd.read_pickle("./rec_game_list.pkl")


###--------------------------------------
#result_R.iloc[0].values[0]
##해당 게임에 대한 유저들 score
#users_choiceGame_score = []
#for i in range(result_R.shape[0]):
#    for j in range(result_R.shape[1]):
#        if result_R.iloc[i].index[j] =='Dota 2':
#            v = result_R.iloc[i].values[j]
#            users_choiceGame_score.append((i, j, v))
#            
#len(users_choiceGame_score)
###--------------------------------------



