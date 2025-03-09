import os
os.environ["CHROMADB_FORCE_PYSQLITE3"] = "1"

import chromadb
import streamlit as st
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
from bs4 import BeautifulSoup
import pandas as pd
from janome.tokenizer import Tokenizer
from rank_bm25 import BM25Okapi

load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

# LLM 初期化
def initialize_llm(apikey, model, temperature, max_tokens):
    llm = ChatOpenAI(api_key=apikey, model=model, temperature=temperature, max_tokens=max_tokens)
    return llm

# Tokenizerの初期化
t = Tokenizer()

# 文書用のTokenizerの定義
def tokenize(text):
    return [token.surface for token in t.tokenize(text)]

# クエリ用のTokenizerの定義
def query_tokenize(text):
    return [token.surface for token in t.tokenize(text) if token.part_of_speech.split(',')[0] in ["名詞", "動詞", "形容詞"]]


# RAG チェーンセットアップ
def setup_chain(llm, vector_store, bm25_retriever):
    vector_retriever = vector_store.as_retriever()
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question in Japanese. "
        "If you don't clearly know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    rag_chain = create_retrieval_chain(
        retriever=vector_retriever,
        combine_docs_chain=question_answer_chain
    )

    return rag_chain, vector_retriever, bm25_retriever


# Streamlit UI
st.header("ドコモビジネスオンラインショップ")
st.subheader("お問い合わせサポートv1.0.0\n"
             "ドコモビジネスオンラインショップの下記ページをもとに質問に回答します。\n"
             "https://www.onlineshop.docomobusiness.ntt.com/guide\n"
             "https://onlineshop.docomobusiness.ntt.com/guide/guest\n"
             "https://www.onlineshop.docomobusiness.ntt.com/guide/user\n")
st.write("※BM25のキーワード検索とベクトル検索のハイブリッド検索（RFFを使用）に基づいて処理しています。")

with st.sidebar:
    st.title("Setting")
    api_key = st.text_input("Please enter your OpenAI API Key", type ="password")
    model = st.selectbox("Select model",['gpt-4o-mini', 'gpt-4o', 'gpt-40-turbo'])

    # Adjust response parameters
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
    max_tokens = st.slider("Max tokens", min_value=100, max_value=500, value=250)


# 処理するURLリスト
urls = [
    "https://www.onlineshop.docomobusiness.ntt.com/guide",
    "https://www.onlineshop.docomobusiness.ntt.com/guide/guest",
    "https://www.onlineshop.docomobusiness.ntt.com/guide/user"
]

# 抽出対象のクラス名のリスト
target_classes = [
    "guide-detail", "faq-list__counter--all",
    "main-heading", "catch-text", "flow-detail",
    "main-heading", "catch-text", "note-box","flow-detail"
]

# text splitterの定義
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=1000,
    chunk_overlap=50
)

# ページ情報の格納リスト
df_splits = pd.DataFrame(columns=["page", "chunk_no", "text"])

# 各URLを処理
for url in urls:
    # ページのHTMLを取得
    response = requests.get(url)
    response.raise_for_status()  # HTTPエラーが発生した場合は例外を発生させる
    html_content = response.text

    # BeautifulSoupでHTMLを解析
    soup = BeautifulSoup(html_content, 'html.parser')

    # 指定した複数のクラス名に一致する要素をすべて取得
    elements = soup.find_all(class_=target_classes)

    # 各要素からテキストを取得し、スペースで結合
    extracted_text = " ".join(element.get_text(strip=True) for element in elements)

    chunk = text_splitter.split_text(extracted_text)
    for i, c in enumerate(chunk):
        df_splits = pd.concat([df_splits, pd.DataFrame({"page": url, "chunk_no": i, "text": c}, index=[0])])

# データ識別用にIDを作成
df_splits["ID"] = df_splits["page"].str.replace("https://www.onlineshop.docomobusiness.ntt.com", "", regex=False) + "-" + df_splits["chunk_no"].astype(str)
df_splits = df_splits[["ID", "page", "chunk_no", "text"]]

# textとmetadataをリスト型で定義
text_list = df_splits["text"].tolist()
metadata_list = df_splits[["ID", "page"]].to_dict(orient="records")


####################### キーワード検索の実装 #######################
# 文書を単語リストに分割
tokenized_documents = [tokenize(doc) for doc in text_list]
# BM25
bm25 = BM25Okapi(tokenized_documents)


######################## ベクトル検索の実装########################
embeddings = HuggingFaceEmbeddings() # Embeddingの定義
client = chromadb.PersistentClient() # PersistentClientの初期化
db = Chroma(
        collection_name="langchain_store",  # 既存コレクションを使用
        embedding_function=embeddings,
        client=client,
    )

collection_data = db.get()



# 質問入力
user_input = st.text_input("質問:")

if user_input and api_key:
    llm = initialize_llm(api_key, model, temperature, max_tokens)
    query = user_input
    n=len(text_list) # チャンク数の計算

    ########## ベクトル検索の実行 ##########
    vector_top = db.similarity_search(query=query, k=n)

    # 詳細情報を取得して表示
    # df_vector_top = pd.DataFrame([doc.page_content for doc in vector_top], columns=["text"])
    vector_rank_list = [{"text": doc.page_content, "vector_rank": i + 1} for i, doc in enumerate(vector_top)]
    df_vector = pd.DataFrame(vector_rank_list)

    # vector_retriever = pd.merge(df_vector, df_splits, on="text", how="left")

    ########## キーワード検索の実行 ##########
    tokenized_query = query_tokenize(query) # クエリをキーワード単語リストに分割して検索
    keyword_top = bm25.get_top_n(tokenized_query, text_list, n=n)

    # 詳細情報を取得して表示
    # df_keyword_top = pd.DataFrame(keyword_top, columns=["text"])
    keyword_rank_list = [{"text": doc, "keyword_rank": i + 1} for i, doc in enumerate(keyword_top)]
    df_keyword = pd.DataFrame(keyword_rank_list)


    ########## ハイブリッド検索の実行 ##########
    df_rank = pd.merge(df_vector, df_keyword, on="text", how="left")
    df_rank["hybrid_score"] = 1 / (df_rank["vector_rank"] + 60) + 1 / (df_rank["keyword_rank"] + 60)
    df_rank = pd.merge(df_rank, df_splits, on="text", how="left")
    df_hybrid_top = df_rank.sort_values(by="hybrid_score", ascending=False).head()


    ########## chainの実行 ##########
    df = df_hybrid_top.reset_index()
    context = df.loc[0, "text"]

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question in Japanese. "
        "If you don't clearly know the answer, say that you don't know. "
        "Use five sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    chain = (prompt|llm)
    response = chain.invoke({"context":context, "input": user_input})

    st.write("---")
    st.write("回答：")
    st.write(response.content)

    # st.write("hybrid_retriever")
    # st.dataframe(df_hybrid_top)
    # st.write("vector_retriever")
    # st.dataframe(df_vector)
    # st.write("keyword_retriever")
    # st.dataframe(df_keyword)
elif user_input:
    st.warning("OpenAI のAPIkeyを入力してください。")
else:
    st.write("質問を入力してください。")
