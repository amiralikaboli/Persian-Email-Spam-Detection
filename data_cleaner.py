import json
import os
import re

import hazm
import parsivar


class TextCleaner:
    def __init__(self):
        self.punctuations = [
            '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '/', ':', ';', '<', '=', '>', '@', '[',
            '\\', ']', '^', '_', '`', '{', '|', '}', '~', '£', '¤', '§', '©', '«', '®', '°', '±', '²', '´', '¸', '»',
            '¼', '½', '¾', '×', '÷', 'ˈ', '˜', '˝', '٪', '٫', '٬', '‐', '–', '—', '‘', '’', '“', '”', '„', '…', '″',
            '‹', '›', '™', '↑', '→', '↓', '⋅', '⌘', '▪', '◄', '○', '♫', '✓', '❤', '《', '》', '爆', '者', '被', '\uf020',
            '\uf04f', '\uf05f', '\uf076', '\uf0a7', '\uf0fc', '﴾', '﴿', '：', '�', '?', '؟', '.', '،', '؛', '•', '●'
        ]
        self.diacritics_pattern = re.compile("[\u064B-\u065e\u0670\u0674\u06c3\u06d4-\u06ed]")
        self.emojis_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            "]+",
            flags=re.UNICODE
        )
        self.latin_characters_pattern = re.compile(
            "["
            "\u0041-\u007a"
            "\u00c0-\u036f"
            "\u0400-\u050f"
            "\u0342-\u03ff"
            "]"
        )
        self.numbers_pattern = re.compile("[0-9]")
        self.space_patterns = [
            (re.compile("[\u202c\u2005\u2009\u2029\u2066\u3000\ufe0f]"), ' '),
            (re.compile("[\f\r\t\n]"), ' '),
            (re.compile("[\u001f\u009d\u200a\u200e\u200f\u206d\xa0\xad]"), '\u200c'),
            (re.compile("[\u007f\u0085\u061c\u200b\u200d\u202a\u202b\u206f\u2003"
                        "\u2028\u2060\u2063\u2067\u2069\ufeff\ufffc\x18]"), ''),
        ]
        self.stopwords = hazm.stopwords_list()[:200] + [
            'ام', 'م', 'ات', 'ای', 'ی', 'ت', 'اش', 'ش', 'مان', 'یم', 'ایم', 'تان', 'ید', 'اید', 'شان', 'ند', 'اند',
            'است', 'هست', 'بود', 'شد', 'شو', 'باش', 'خواه', 'ها', 'های', 'ان', 'یک', 'دو', 'سه', 'چهار', 'پنج', 'شش',
            'هفت', 'هشت', 'نه', 'ده', 'هستم', 'هستم', 'هست', 'هستید', 'هستیم', 'نیستم', 'نیستی', 'نیست', 'نیستیم',
            'نیستید', 'نیستند'
        ]

        self.normalizer = parsivar.Normalizer()
        self.stemmer = parsivar.FindStems()
        self.lemmatizer = hazm.Lemmatizer()

    def remove_punctuations(self, text: str) -> str:
        for punctuation in self.punctuations:
            text = text.replace(punctuation, '')
        return text

    def remove_diacritics(self, text: str) -> str:
        return self.diacritics_pattern.sub(r'', text)

    def remove_emojis(self, text: str) -> str:
        return self.emojis_pattern.sub(r'', text)

    def remove_latin_characters(self, text: str) -> str:
        return self.latin_characters_pattern.sub(r'', text)

    def remove_numbers(self, text: str) -> str:
        return self.numbers_pattern.sub('', text)

    def unify_spaces(self, text: str) -> str:
        for pattern, repl in self.space_patterns:
            text = pattern.sub(repl, text)
        text = text.replace('  ', ' ')
        return text

    def remove_stopwords_and_stem(self, text):
        tokens = text.split()

        final_tokens = []
        for token in tokens:
            stemmed_token = self.stemmer.convert_to_stem(self.lemmatizer.lemmatize(token)).replace('&', '#')
            if '#' in stemmed_token:
                past, present = stemmed_token.split('#')
                stemmed_token = past if past in token else present
            if token not in self.stopwords and stemmed_token not in self.stopwords:
                final_tokens.append(stemmed_token)

        return ' '.join(final_tokens)

    def clean_text(self, text: str) -> str:
        text = self.normalizer.sub_alphabets(text)
        text = self.remove_latin_characters(text)
        text = self.remove_numbers(text)
        text = self.remove_punctuations(text)
        text = self.remove_diacritics(text)
        text = self.remove_emojis(text)
        text = self.unify_spaces(text)
        text = self.normalizer.space_correction(text)

        return text

    def clean_dataset(self):
        train_data = []
        test_data = []
        for ham_or_spam in ['ham', 'spam']:
            for train_or_test in ['training', 'testing']:
                dir_path = os.path.join(os.path.dirname(__file__), 'data', f'{ham_or_spam}{train_or_test}')
                for file_name in os.listdir(dir_path):
                    with open(f'{dir_path}/{file_name}', 'r') as txt_file:
                        text = txt_file.read()
                        text = self.clean_text(text)
                        text = self.remove_stopwords_and_stem(text)

                        if train_or_test == 'training':
                            train_data.append((text, ham_or_spam))
                        else:
                            test_data.append((text, ham_or_spam))

        with open('data/train_data.json', 'w') as json_file:
            json.dump(train_data, json_file, ensure_ascii=False)
        with open('data/test_data.json', 'w') as json_file:
            json.dump(test_data, json_file, ensure_ascii=False)


if __name__ == '__main__':
    text_cleaner = TextCleaner()
    text_cleaner.clean_dataset()
