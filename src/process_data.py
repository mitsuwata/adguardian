import pandas as pd

def select_columns(df):
    selected_columns = ['違反有無', '給与Index', '給与', '時間', '休日', '所在地と勤務地', '就業形態1', '就業形態2']
    df2 = df[selected_columns].copy()  # Copy the selected columns to a new DataFrame
    
    df2.loc[:, '違反有無'] = df2['違反有無'].apply(lambda x: 0 if pd.isna(x) else 1 if x == '違反' else x)
    df3 = df2.fillna('未記入')
    df4 = df3.assign(テキスト=df3['給与Index'].astype(str) + df3['給与'].astype(str) + df3['時間'].astype(str) + df3['休日'].astype(str) + df3['所在地と勤務地'].astype(str) + df3['就業形態1'].astype(str) + df3['就業形態2'].astype(str))
    selected_df = df4[['違反有無', 'テキスト']]
    return selected_df
