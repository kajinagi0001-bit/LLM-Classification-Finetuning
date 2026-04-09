# LLM-Classification-Finetuning

4/01 Baseline
4/02 src 作成
4/09 taniguchi-space:TokenAttentionPooling_Layer作成  

要確認：
BASE_PATHあっているか
csvデータはこのディレクトリにないのでinputファイルの中に個別で入れる必要あり

実行：
python src/main.py --wandb --mode (train/infer/test)
wandb使わないときは宣言なし


環境:
Docker cuda+pytorchでerror出る可能性あり
Dockerコマンド
- コンテナの作成・起動:  docker compose up
- コンテナの一覧表示:  docker compose ps   (-a で停止中のものも表示)
- コンテナでコマンド実行:  docker exec -it "コンテナ名" /bin/bash
- コンテナの停止: docker compose down

次はforward時にautocastを使用して動かす。（コード修正済みexp003_lr1e4_autocast）
autocast使わないときとの精度比較
うまくいけば、データサイズ戻す
autocast単体で使うと学習されない
gradscalerとの併用が前提？

lr:
1e-4　val_accが上がらない、過学習
1e-5  3エポックまではval_acc上昇0.46程度
1e-6 