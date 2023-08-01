import pandas as pd

import io
from flask import Flask, render_template, request, make_response, send_file
import csv
import tempfile
import process_data

import torch
import torch.nn.functional as F
import transformers
from transformers import AlbertTokenizer
from transformers import AlbertForSequenceClassification

# 学習済みモデルをもとに推論する
def predict(df):
    # Tokenizerの準備
    model_name = 'ALINEAR/albert-japanese-v2'
    albert_tokenizer = AlbertTokenizer.from_pretrained(model_name)

    # テキストデータの取り出し
    x_text = df['テキスト'].tolist()

    # 学習済みモデルの読み込み
    new_model = AlbertForSequenceClassification.from_pretrained('./model')

    # 結果を格納するためのリスト
    results = []

    max_length = 512
    with torch.no_grad():

    # 各テキストに対してループで推論を行う
        for text in x_text:
            inputs = albert_tokenizer(text, return_tensors="pt", max_length=max_length, padding='max_length', truncation=True)
            output = new_model(**inputs)

            # ソフトマックス関数を適用して確率分布に変換
            probs = F.softmax(output.logits, dim=1)

            # 最大の確率に対応するラベルを取得
            predicted_labels = torch.argmax(probs, dim=1)

            # 確率分布からラベル1に対応する確率を取得
            prob_label_1 = probs[:, 1]

            # 0.3以上なら1のラベルを付ける
            #predicted_labels[prob_label_1 >= 0.3] = 1

            # 結果を辞書にまとめてリストに追加
            result_dict = {
                "text": text,
                "logits": output.logits,
                "predicted_labels": predicted_labels,
                "prob_label_1": prob_label_1
            }
            results.append(result_dict)
    
    predicted_labels_list = [item['predicted_labels'].item() for item in results]
    prob_label_1_list = [item['prob_label_1'].item() for item in results]

    # 新しい列のデータ
    new_columns = {
        '予測': predicted_labels_list,
        '確率': prob_label_1_list
    }

    # 新しい列をDataFrameに一度に追加
    df_complete = df.assign(**new_columns)

    return df_complete



app = Flask(__name__)

csv_output = None

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/upload', methods=['POST'])
def upload():
    global csv_output

    file = request.files['file']
    if not file:
        return 'ファイルアップロードされていません.', 400
    if file.filename.endswith('.csv'):
        csv_file = file.stream.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(csv_file))

        # 列を選択してDataFrameを作成
        selected_df = process_data.select_columns(df)
        #print(selected_df)

        df_complete = predict(selected_df)

        csv_output = df_complete.to_csv(index=False)  # DataFrameをCSV形式の文字列に変換
        return render_template('table.html', csv_output=csv_output)
    else:
        return 'CSVファイルではありません.', 400

@app.route('/download', methods=['POST'])
def download():
    global csv_output

    if not csv_output:
        return 'CSVデータがありません.', 400

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(csv_output.encode("utf-8"))
    temp_file.close()

    return send_file(
        temp_file.name,
        mimetype='text/csv',
        as_attachment=True,
        download_name='data.csv'
    )

if __name__ == '__main__':
    app.run(debug=True)