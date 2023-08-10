import pandas as pd

import io
from flask import Flask, render_template, request, make_response, send_file
from flask_socketio import SocketIO

import csv
import tempfile
import process_data

import torch
import torch.nn.functional as F
import torch.quantization
import transformers
from transformers import AlbertTokenizer
from transformers import AlbertForSequenceClassification

# 学習済みモデルをもとに推論する
def predict(df,data_count):
    # Tokenizerの準備
    model_name = 'ALINEAR/albert-japanese-v2'
    albert_tokenizer = AlbertTokenizer.from_pretrained(model_name)

    # テキストデータの取り出し
    x_text = df['テキスト'].tolist()

    # 学習済みモデルの読み込み
    new_model = AlbertForSequenceClassification.from_pretrained('./model')

    # モデルを量子化
    quantized_model = torch.quantization.quantize_dynamic(
        new_model,  # 学習済みモデル
        {torch.nn.Linear, torch.nn.Conv2d},  # 量子化するモジュールのリスト
        dtype=torch.qint8  # 量子化データ型
    )

    # 結果を格納するためのリスト
    quantized_results = []

    max_length = 266
    with torch.no_grad():

    # 各テキストに対してループで推論を行う
        for idx, text in enumerate(x_text):
            inputs = albert_tokenizer(text, return_tensors="pt", max_length=max_length, padding='do_not_pad', truncation=True)
            print('符号化が終わった')
            quantized_output = quantized_model(**inputs)
            print('ベクトル化が終わった')

            # ソフトマックス関数を適用して確率分布に変換
            probs = F.softmax(quantized_output.logits.float(), dim=1)

            # 最大の確率に対応するラベルを取得
            predicted_labels = torch.argmax(probs, dim=1)

            # 確率分布からラベル1に対応する確率を取得
            prob_label_1 = probs[:, 1]

            # 0.3以上なら1のラベルを付ける
            #predicted_labels[prob_label_1 >= 0.3] = 1

            # 結果を辞書にまとめてリストに追加
            result_dict = {
                "text": text,
                "logits": quantized_output.logits,
                "predicted_labels": predicted_labels,
                "prob_label_1": prob_label_1
            }
            quantized_results.append(result_dict)
            print(f'推論が終わった: {idx+1}/{data_count}')
            if idx % 10 == 0:
                socketio.emit('update_letters', {'letters': idx})
    
    predicted_labels_list = [item['predicted_labels'].item() for item in quantized_results]
    prob_label_1_list = [item['prob_label_1'].item() for item in quantized_results]

    # 新しい列のデータ
    new_columns = {
        '予測': predicted_labels_list,
        '確率': prob_label_1_list
    }

    # 新しい列をDataFrameに一度に追加
    df_complete = df.assign(**new_columns)

    return df_complete



app = Flask(__name__)
socketio = SocketIO(app, ping_interval=10, ping_timeout=5)

csv_output = None
selected_df = None
data_count = None

@app.route('/')
def home():
    return render_template('home.html')

@socketio.on('connect')
def handle_connect():
    socketio.emit('update_letters2', {'letters2': '繋がった！'})

@app.route('/upload', methods=['POST'])
def upload():
    global csv_output
    global selected_df
    global data_count  
    file = request.files['file']
    if not file:
        return 'ファイルアップロードされていません.', 400
    if file.filename.endswith('.csv'):
        csv_file = file.stream.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(csv_file))

        # 列を選択してDataFrameを作成
        selected_df, data_count = process_data.select_columns_fast(df)
        print(f'データができた。データの件数: {data_count}')

        return render_template('progress.html',data_count=data_count)
    else:
        return 'CSVファイルではありません.', 400

@socketio.on('start_generating')
def start_generating():
    print('start_generating')
    socketio.emit('update_letters', {'letters': '処理中'})
    global csv_output
    global selected_df
    global data_count  
    df_complete = predict(selected_df,data_count)
    csv_output = df_complete.to_csv(index=False)  # DataFrameをCSV形式の文字列に変換
    # フロントエンドに処理完了のイベントを送信
    socketio.emit('generation_complete')
    return

@app.route('/table')
def table():
    return render_template('table.html')


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
    socketio.run(app,debug=True)