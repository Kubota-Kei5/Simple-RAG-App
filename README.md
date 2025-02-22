# Simple-RAG-App
RFFを使ったハイブリッド検索によるRAGのQAアプリ<br>
<br>
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
・類似度が最も高い文書を使ってLLMに回答を要求<br>
<br>
今後の改善ポイント：<br>
1. Improved processing speed<br>
質問入力から回答出力までの時間を短縮する。<br>
~~アプリ起動のたびにソースデータの読み込み～Embeddings～vectorstore作成を行っているため、事前にベクトルDBを作成しておくことで処理速度を短縮。~~(修正済み：2025/02/22)<br>
<br>
2. Improved text pre-processing accuracy<br>
ソースデータの前処理の精度を向上させる。記号などを正しく削除する割合を増やす。また、「詳しくはこちら」という記載の直後にurlを挿入することで回答文の使いやすさ向上も実現したい。<br>
<br>
<br>
3. Improved text pre-processing accuracy<br>
ソースデータの前処理の精度を向上させる。記号などを正しく削除する割合を増やす。また、詳しくはこちらという記載の直後にurlを挿入することで回答の精度向上も狙う。<br>
<br>
4. Result Re-Rank<br>
マイクロソフトのSearchで述べられている通り、ハイブリッド検索とリランクモデルを併用することで精度向上が期待できる。<br>
https://techcommunity.microsoft.com/blog/azure-ai-services-blog/azure-ai-search-outperforming-vector-search-with-hybrid-retrieval-and-reranking/3929167<br>
<br>
5. HyDE<br>
HyDE（Hypothetical Document Embeddings）とは質問文のみでLLMから仮回答を生成し、その仮回答をEmbeddingして類似度検索を行うというもの。<br>
<br>
6. Metadata Attachments<br>
メタデータを追加しまくる。ここは単純にデータをいっぱい準備するだけ。趣味の範囲でやっているので着手するモチベーションは低い。<br>
<br>
7. knowledge Graphs<br>
1~5で満足のいく精度が得られなかった場合に手を出す。ナレッジグラフ用のDB（Neo4j）を準備する必要があるためハードルが高い。Neo4jはCypherとかいうクエリ言語の習熟も必要。CypherのクエリをGPTに書いてもらうという方法もある
