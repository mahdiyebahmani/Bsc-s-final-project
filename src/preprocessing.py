import re
import unicodedata

from src.utils import PROCESSED_DATA_DIR

from hazm import (
    stopwords_list,
    Lemmatizer,
    Stemmer,
    word_tokenize,
    Normalizer
)


class PersianPreprocessor:

    def __init__(self):
        self.normalizer = Normalizer()
        self.lemmatizer = Lemmatizer()
        self.stemmer = Stemmer()
        self.stop_words = set(stopwords_list())
        self.protected_words = {
            'خوب', 'خوبی', 'عالی', 'بهتر', 'بهترین', 'مناسب', 'مهم',
            'متفاوت', 'زیاد', 'کم', 'کامل', 'کاملا', 'قابل', 'بدون',
            'فقط', 'خیلی', 'بسیار', 'متاسفانه',
            'نه', 'نیست', 'نیستند', 'نبود', 'نباید', 'ندارد', 'ندارند',
            'نمی\u200cشود'}

        self.stop_words -= self.protected_words
        self.html_tags = re.compile(r'<.*?>')
        self.url_pattern = re.compile(r'http\S+|www\.\S+')
        self.mention_pattern = re.compile(r'@\w+')
        self.punctuations = re.compile(
            r"[!\"$%&'()*+,\-./:;<=>?@\[\]^`{|}~،؛؟«»٪]")

        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002700-\U000027BF"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE)

        self.extra_spaces = re.compile(r'\s+')

        self.special_chars = re.compile(
            r"[*/&|<>~+=\\^™%\"”“❝„]+")

        self.unwanted_chars = re.compile(
            r"[【】{}()\[\]‼…]+")

    def word_remove(self, text):
        text = self.url_pattern.sub(" ", text)
        text = self.mention_pattern.sub(" ", text)
        text = re.sub(r'#', '', text)
        text = re.sub(r'_', ' ', text)
        return text

    def char_removing(self, text):
        text = self.unwanted_chars.sub(" ", text)
        text = self.special_chars.sub(" ", text)
        text = self.remove_unicode_controls(text)
        return text

    def word_stopwords(self, text):
        tokens = word_tokenize(text)

        return ' '.join(
            word for word in tokens
            if word not in self.stop_words
        )

    def lemmatization(self, text):
        tokens = word_tokenize(text)
        lemmas = []

        for word in tokens:
            lemma = self.lemmatizer.lemmatize(word)

            if "#" in lemma:
                lemma = lemma.split("#")[0]

            lemmas.append(lemma)

        return " ".join(lemmas)

    def stemming(self, text):
        tokens = word_tokenize(text)

        return " ".join(
            self.stemmer.stem(token)
            for token in tokens
        )

    def remove_emojis(self, text):
        return self.emoji_pattern.sub('', text)

    def remove_unicode_controls(self, text):
        return ''.join(
            char
            for char in text
            if unicodedata.category(char) not in {'Cc', 'Cf'}
            or char == '\u200c'
        )

    def clean_text(self, text):
        text = "" if text is None else str(text)
        text = re.sub(r'_x000D_', ' ', text, flags=re.IGNORECASE)
        text = self.normalizer.normalize(text)
        text = self.html_tags.sub(' ', text)
        text = self.word_remove(text)
        text = self.remove_emojis(text)
        text = text.replace('٪', ' درصد ')
        text = self.punctuations.sub(' ', text)
        text = self.char_removing(text)
        text = self.extra_spaces.sub(" ", text)
        text = self.word_stopwords(text)
        text = self.extra_spaces.sub(" ", text).strip()
        return text


def preprocess_dataframe(df):
    clean_lemma_path = PROCESSED_DATA_DIR / "clean_lemma.csv"
    clean_stem_path = PROCESSED_DATA_DIR / "clean_stem.csv"

    processor = PersianPreprocessor()

    df["text_clean"] = (
        df["text"]
        .fillna("")
        .map(processor.clean_text)
    )

    df["text_lemma"] = (
        df["text_clean"]
        .map(processor.lemmatization)
    )

    df["text_stem"] = (
        df["text_clean"]
        .map(processor.stemming)
    )

    clean_lemma_df = df[
        [column for column in df.columns if column not in ["text_stem"]]
    ]

    clean_lemma_df.to_csv(
        clean_lemma_path,
        index=False,
        encoding="utf-8-sig"
    )

    clean_stem_df = df[
        [column for column in df.columns if column not in ["text_lemma"]]
    ]

    clean_stem_df.to_csv(
        clean_stem_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nSaved clean + lemma")
    print("Saved clean + stem")

    return clean_lemma_df, clean_stem_df