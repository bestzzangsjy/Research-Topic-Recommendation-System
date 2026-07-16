import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(page_title="고교 세특 주제 추천기", layout="centered")

@st.cache_resource
def load_model():
    return SentenceTransformer('jhgan/ko-sroberta-multitask')

@st.cache_data
def load_data():
    df = pd.read_csv('topics.csv', encoding='utf-8')
    df['combined_text'] = df['category'] + " " + df['subject'] + " " + df['topic'] + " " + df['description'] + " " + df['keywords']
    return df

model = load_model()
df = load_data()

st.title("🎓 맞춤형 세특 탐구주제 추천기")
st.write("관심 있는 분야나 융합하고 싶은 과목을 입력하면 딱 맞는 탐구 주제를 찾아줍니다.")

user_query = st.text_input("🔍 검색어를 입력하세요 (예: 물리와 인공지능 융합)", "")

if st.button("추천 받기") and user_query:
    with st.spinner("알맞은 탐구 주제를 찾는 중입니다..."):
        dataset_embeddings = model.encode(df['combined_text'].tolist())
        query_embedding = model.encode([user_query])
        
        similarities = cosine_similarity(query_embedding, dataset_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:3]
        
        st.success("너에게 딱 맞는 탐구 주제를 찾았어!")
        st.markdown("---")
        
        for rank, idx in enumerate(top_indices, 1):
            row = df.iloc[idx]
            score = similarities[idx]
            
            with st.container():
                st.subheader(f"{rank}등. [{row['category']}] {row['topic']}")
                st.write(f"**관련 교과:** {row['subject']}")
                st.write(f"**탐구 내용:** {row['description']}")
                st.markdown("---")
