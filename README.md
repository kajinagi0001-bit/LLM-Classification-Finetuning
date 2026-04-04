# LLM-Classification-Finetuning

4/01 Baseline
4/02 src 作成

要確認：
BASE_PATHあっているか
csvデータはこのディレクトリにないのでinputファイルの中に個別で入れる必要あり

実行：
src/main.py


環境:
Docker cuda+pytorchでerror出る可能性あり
Dockerコマンド
- コンテナの作成・起動:  docker compose up
- コンテナの一覧表示:  docker compose ps   (-a で停止中のものも表示)
- コンテナでコマンド実行:  docker exec -it "コンテナ名" /bin/bash
- コンテナの停止: docker compose down
