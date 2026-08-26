from src.utils import RAW_DATA_DIR, PROCESSED_DATA_DIR
import pandas as pd


def load_data(file_path= RAW_DATA_DIR/"keifiat.csv"):
    df = pd.read_csv(file_path, encoding='utf-8-sig')

    print(df.columns.to_list())
    print(df.shape)

    df = df.drop(columns=[
        'product_id',
        'product_title',
        'title_en',
        'user_id',
        'likes',
        'dislikes',
        'verification_status'
    ])

    print(df.columns.to_list())

    print(df['recommend'].unique())

    labels = [
        'recommended',
        'not_recommended',
        'no_idea'
    ]

    df = df[df['recommend'].isin(labels)].copy()

    label_map = {
        'not_recommended': 0,
        'no_idea': 1,
        'recommended': 2
    }

    df['label'] = df['recommend'].map(label_map)

    cols = [
        'title',
        'comment',
        'advantages',
        'disadvantages'
    ]

    for col in cols:
        df[col] = df[col].fillna('').astype(str)

    df['text'] = (
        df['title'] + ' ' +
        df['comment'] + ' ' +
        df['advantages'] + ' ' +
        df['disadvantages']
    )

    df['text'] = (
        df['text']
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )

    print(df['text'].head())
    print(df['label'].value_counts().sort_index())
    df[['text', 'label']].to_csv(
    PROCESSED_DATA_DIR / 'parsbert_data.csv',
    index=False,
    encoding='utf-8-sig'
)

    return df