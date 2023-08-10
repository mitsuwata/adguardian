import pandas as pd

def select_columns_fast(df):
    selected_columns = ['違反有無', '給与Index', '給与', '時間', '休日', '所在地と勤務地', '就業形態1', '就業形態2']
    df2 = df[selected_columns].copy()

    # 違反有無列を数値に変換
    df2['違反有無'] = df2['違反有無'].apply(lambda x: 0 if pd.isna(x) else 1 if x == '違反' else x)

    # 欠損値を '未記入' に置き換え
    df2.fillna('未記入', inplace=True)

    # テキスト列を結合
    df2['テキスト'] = (df2['給与Index'].astype(str) + df2['給与'].astype(str) + df2['時間'].astype(str) +
                     df2['休日'].astype(str) + df2['所在地と勤務地'].astype(str) + df2['就業形態1'].astype(str) +
                     df2['就業形態2'].astype(str))

    # 選択した列のみ取得
    selected_df = df2[['違反有無', 'テキスト']]

    # データの件数を戻り値として返す
    data_count = len(selected_df)

    return selected_df, data_count

