# Simple-RAG-App
RFFを使ったハイブリッド検索によるRAGのQAアプリ

使い方：<br>
・OpenAIのAPIkeyをSettingに入力<br>
・LLMのモデルを選択<br>
・temperature,max_tokensを設定<br>
・質問を入力<br>
・回答が出力される<br>
<br>
処理方法<br>
・事前に3つのURLをスクレイピングしてvectorstoreを作成（Chromadb）<br>
・BM25のキーワード検索と事前に作成していたvectorstoreを使ったベクトル検索で類似文書を抽出<br>
・RFFを使って文書の類似度を調整<br>
・類似度が最も高い文書を使ってLLMに回答を要求
