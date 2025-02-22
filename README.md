# Simple-RAG-App
RFFを使ったハイブリッド検索によるRAGのQAアプリ

使い方：
・OpenAIのAPIkeyをSettingに入力
・LLMのモデルを選択
・temperature,max_tokensを設定
・質問を入力
・回答が出力される

処理方法
・事前に3つのURLをスクレイピングしてvectorstoreを作成（Chromadb）
・BM25のキーワード検索と事前に作成していたvectorstoreを使ったベクトル検索で類似文書を抽出
・RFFを使って文書の類似度を調整
・類似度が最も高い文書を使ってLLMに回答を要求
