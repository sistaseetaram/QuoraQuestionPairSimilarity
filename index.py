# %%
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle as pk
# from ipynb.fs.full.Text_vectorization_model_selection import apply_
from urllib.request import urlopen
import joblib as jb
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import distance
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
from scipy.sparse import hstack
from tqdm import tqdm
import base64

# from EDA_ON_QQPS import Preprocessor
# %%
# DATA_URL = ('E:/')
os.chdir('E:/Projects/QuoraQuestionPairSimilarity/')
# %%
# @st.cache
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('background.png')

def load_data():
    # model = pk.load(urlopen(DATA_URL, 'rb'))    
    df=pd.read_pickle("nlp_final_train.pkl")
    with open('tfidf.pkl','rb') as txt_pk:
        tfidf = pk.load(txt_pk)
    with open('Stack.pkl','rb') as pk_file:
        model = pk.load(pk_file)
    return df,model,tfidf

data_load_state =st.text('Loading Required dependencies...')

df,model,tfidf = load_data()

data_load_state.text("Model is ready!")

# %%
import re
def extract_unique(data):
    t1 = data[["qid1","question1"]]
    t2 = data[["qid2","question2"]]
    t1.columns =t2.columns=["qid","question"]
    new = pd.concat([t1,t2],ignore_index=True)
    unique_ques = new.drop_duplicates(subset="qid")
    return unique_ques,new
# %%
def preprocess(x):
    '''This fuction is used to clan data and replace text with some common conventions used in emperical use of language.'''
    x = str(x).lower()
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")                               .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")                               .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")                               .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")                               .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")                               .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")                               .replace("€", " euro ").replace("'ll", " will")
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)



    lemmatizer = WordNetLemmatizer()
    pattern = re.compile('\W')

    if isinstance(x,str):
        x = re.sub(pattern, ' ', x)    
    if isinstance(x,str):
        x = ' '.join(list(map(lambda y: lemmatizer.lemmatize(y),x.split(' ')))) 
        example1 = BeautifulSoup(x)
        x = example1.get_text()
    return x.strip()

# %%
class Featurizer: 
    def __init__(self): 
        self.SAFE_DIV=0.0001
        self.STOP_WORDS = stopwords.words("english")
        self.total = total_ques
        self.unique = unique_ques
        self.grp_by_qid = total_ques.groupby("qid").count()
    c=0
    def get_token_features(self,q1, q2):
        token_features = [0.0]*10
        # Converting the Sentence into Tokens: 
        q1_tokens = q1.split()
        q2_tokens = q2.split()

        if len(q1_tokens) == 0 or len(q2_tokens) == 0:
            return token_features
        # Get the non-stopwords in Questions
        q1_words = set([word for word in q1_tokens if word not in self.STOP_WORDS])
        q2_words = set([word for word in q2_tokens if word not in self.STOP_WORDS])

        #Get the stopwords in Questions
        q1_stops = set([word for word in q1_tokens if word in self.STOP_WORDS])
        q2_stops = set([word for word in q2_tokens if word in self.STOP_WORDS])

        # Get the common non-stopwords from Question pair
        common_word_count = len(q1_words.intersection(q2_words))

        # Get the common stopwords from Question pair
        common_stop_count = len(q1_stops.intersection(q2_stops))

        # Get the common Tokens from Question pair
        common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))


        token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + self.SAFE_DIV)
        token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + self.SAFE_DIV)
        token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + self.SAFE_DIV)
        token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + self.SAFE_DIV)
        token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + self.SAFE_DIV)
        token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + self.SAFE_DIV)

        # Last word of both question is same or not
        token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])

        # First word of both question is same or not
        token_features[7] = int(q1_tokens[0] == q2_tokens[0])

        token_features[8] = abs(len(q1_tokens) - len(q2_tokens))

        #Average Token Length of both Questions
        token_features[9] = (len(q1_tokens) + len(q2_tokens))/2
        return token_features

    # get the Longest Common sub string
    def get_basic_features(self,qid1,q1,qid2,q2):

        base_features = [0.0]*11
        base_features[0] = self.c
        self.c+=1
        base_features[1] = qid1
        base_features[2] = qid2
        base_features[3] = q1
        base_features[4] = q2
        try:
            base_features[5] = self.grp_by_qid.loc[qid1][0]
        except:
            base_features[5]=1
        try:
            base_features[6] = self.grp_by_qid.loc[qid2][0]
        except:
            base_features[6] =1

        w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), q2.split(" "))) 
        base_features[7] = 1.0 * (len(w1) + len(w2))
        base_features[8] = 1.0 * len(w1 & w2)/(len(w1) + len(w2))

        base_features[9] = base_features[5]+base_features[6]
        base_features[10] = abs(base_features[5]-base_features[6]) 
        return base_features

    def get_longest_substr_ratio(self,a, b):
        strs = list(distance.lcsubstrings(a, b))
        if len(strs) == 0:
            return 0
        else:
            return len(strs[0]) / (min(len(a), len(b)) + 1)

    def extract_features(self,test):
        data = {}
        print("basic features....")
        # base_features = jb.Parallel(n_jobs=8,verbose=2)(jb.delayed(self.get_basic_features)(test['qid'][0],test['question'][0],row[0],row[1]) for row in self.unique.values)
        base_features=[]
        st.text("computing basic features...")
        prg=st.progress(0.0)
        cnt=0
        for row in tqdm(self.unique.values):
            base_features.append(self.get_basic_features(test['qid'][0],test['question'][0],row[0],row[1]))
            cnt+=1
            prg.progress(cnt/self.unique.shape[0])

        data["id"]       = list(map(lambda x: x[0], base_features))
        data["qid1"]       = list(map(lambda x: x[1], base_features))
        data["qid2"]       = list(map(lambda x: x[2], base_features))
        data["question1"]       = list(map(lambda x: x[3], base_features))
        data["question2"]       = list(map(lambda x: x[4], base_features))
        data["freq_qid1"]       = list(map(lambda x: x[5], base_features))
        data["freq_qid2"]       = list(map(lambda x: x[6], base_features))
        data["word_Total"]       = list(map(lambda x: x[7], base_features))
        data["word_share"]       = list(map(lambda x: x[8], base_features))
        data["freq_q1+q2"]       = list(map(lambda x: x[9], base_features))
        data["freq_q1-q2"]       = list(map(lambda x: x[10], base_features))

        print("token features...")

        # Merging Features with dataset
        # token_features = jb.Parallel(n_jobs=4,verbose=2)(jb.delayed(self.get_token_features)(test['question'][0],q2[1]) for q2 in self.unique.values)
        token_features=[]
        st.text("computing token features...")
        prg=st.progress(0.0)
        cnt=0
        for row in tqdm(self.unique.values):
            token_features.append(self.get_token_features(test['question'][0],row[1]))
            cnt+=1
            prg.progress(cnt/self.unique.shape[0])



        data["cwc_min"]       = list(map(lambda x: x[0], token_features))
        data["cwc_max"]       = list(map(lambda x: x[1], token_features))
        data["csc_min"]       = list(map(lambda x: x[2], token_features))
        data["csc_max"]       = list(map(lambda x: x[3], token_features))
        data["ctc_min"]       = list(map(lambda x: x[4], token_features))
        data["ctc_max"]       = list(map(lambda x: x[5], token_features))
        data["last_word_eq"]  = list(map(lambda x: x[6], token_features))
        data["first_word_eq"] = list(map(lambda x: x[7], token_features))
        data["abs_len_diff"]  = list(map(lambda x: x[8], token_features))
        data["mean_len"]      = list(map(lambda x: x[9], token_features))

        #Computing Fuzzy Features and Merging with Dataset

        print("fuzzy features..")
        with st.spinner(text="computing fuzzy features"):
            temp=[]
            st.text("computing token set ratio features...")
            prg=st.progress(0.0)
            cnt=0
            for row in tqdm(self.unique.values):
                temp.append(fuzz.token_set_ratio(test['question'][0],row[1]))
                cnt+=1
                prg.progress(cnt/self.unique.shape[0])
            data["token_set_ratio"]       = temp # jb.Parallel(n_jobs=4,verbose=2)(jb.delayed(fuzz.token_set_ratio)(test['question'][0],q2[1]) for q2 in self.unique.values)
            
            temp=[]
            st.text("computing token set ratio features...")
            prg=st.progress(0.0)
            cnt=0
            for row in tqdm(self.unique.values):
                temp.append(fuzz.token_sort_ratio(test['question'][0],row[1]))
                cnt+=1
                prg.progress(cnt/self.unique.shape[0])
            data["token_sort_ratio"]       = temp # jb.Parallel(n_jobs=4,verbose=2)(jb.delayed(fuzz.token_sort_ratio)(test['question'][0],q2[1]) for q2 in self.unique.values)
            
            temp=[]
            st.text("computing token set ratio features...")
            prg=st.progress(0.0)
            cnt=0
            for row in tqdm(self.unique.values):
                temp.append(fuzz.QRatio(test['question'][0],row[1]))
                cnt+=1
                prg.progress(cnt/self.unique.shape[0])
            data["fuzz_ratio"]       = temp # jb.Parallel(n_jobs=4,verbose=2)(jb.delayed(fuzz.QRatio)(test['question'][0],q2[1]) for q2 in self.unique.values)
          
            temp=[]
            st.text("computing token set ratio features...")
            prg=st.progress(0.0)
            cnt=0
            for row in tqdm(self.unique.values):
                temp.append(fuzz.partial_ratio(test['question'][0],row[1]))
                cnt+=1
                prg.progress(cnt/self.unique.shape[0])
            data["fuzz_partial_ratio"]    = temp #jb.Parallel(n_jobs=4,verbose=2)(jb.delayed(fuzz.partial_ratio)(test['question'][0],q2[1]) for q2 in self.unique.values)
            
            temp=[]
            st.text("computing token set ratio features...")
            prg=st.progress(0.0)
            cnt=0
            for row in tqdm(self.unique.values):
                temp.append(self.get_longest_substr_ratio(test['question'][0],row[1]))
                cnt+=1
                prg.progress(cnt/self.unique.shape[0])
            data["longest_substr_ratio"]  = temp  #jb.Parallel(n_jobs=4,verbose=2)(jb.delayed(self.get_longest_substr_ratio)(test['question'][0],q2[1]) for q2 in self.unique.values)

        return pd.DataFrame(data)

# %%
def finalize(data):
    a = tfidf.transform(data["question1"])
    b = tfidf.transform(data["question2"])
    final = hstack((a,b,data.drop(["question1","question2"],axis=1))).tocsr()
    return final
# %%

def run(ques):
    st.info("Your question is recorded.")
    st.write(ques)
    ques_new = preprocess(ques)
    test = pd.DataFrame({'question':ques_new},index=[0])


    for id,txt in unique_ques.values:
        if txt==test["question"].values[0]:
            test["qid"]=id
            break
    else:
        test["qid"] = unique_ques.shape[0]+1

    with st.spinner(text='Generating Feature Set..'):
        featurize = Featurizer()
        test=featurize.extract_features(test)
    st.success("Features For Final Model Generated")

    final = finalize(test)  

    y_pred= model.predict_proba(final)[:,1]

    indices = np.argsort(y_pred)[-5:]
    top_5_proba = y_pred[indices]
    top_5 = unique_ques.iloc[indices,1]

    disp=list(zip(top_5_proba,top_5))
    with st.expander("View similarity details"):
        st.success('{} Similar questions found!'.format(len(disp)))
        st.write(disp)
        st.line_chart(np.sort(y_pred))
        st.text("Similarity of question asked with all questions in our database...")

# %%
unique_ques,total_ques = extract_unique(df)
total_ques.sort_values(by="qid",inplace=True,ignore_index=True)
unique_ques.sort_values(by="qid",inplace=True,ignore_index=True)
st.header("Quora Question Pair Similarity Prototype")
ques = st.text_area("",value= "Ask Question here...")
st.button("Ask Question.",on_click=run,args=(ques,))
