# повы вклад +
# повы ипотек -
# повы кредит -
# сни ипотек +
# сни кредит +
# санкц -
# как (можно/легко) (взять/узнать/получить/оформить) 0
# >http 0
# при t.co tf-idf высокий. значит надо исключить слова с ними
# парсер городов
# 541 ?

from lxml import etree
from typing import List, Tuple


def load_sentirueval_2016(file_name: str) -> Tuple[List[str], List[str]]:
    texts = []
    labels = []
    with open(file_name, mode='rb') as fp:
        xml_data = fp.read()
    root = etree.fromstring(xml_data)
    for database in root.getchildren():
        if database.tag == 'database':
            for table in database.getchildren():
                if table.tag != 'table':
                    continue
                new_text = None
                new_label = None
                for column in table.getchildren():
                    if column.get('name') == 'text':
                        new_text = str(column.text).strip()
                        if new_label is not None:
                            break
                    elif column.get('name') not in {'id', 'twitid', 'date'}:
                        if new_label is None:
                            label_candidate = str(column.text).strip()
                            if label_candidate in {'0', '1', '-1'}:
                                new_label = 'negative' if label_candidate == '-1' else \
                                    ('positive' if label_candidate == '1' else 'neutral')
                                if new_text is not None:
                                    break
                if (new_text is None) or (new_label is None):
                    raise ValueError('File `{0}` contains some error!'.format(file_name))
                texts.append(new_text)
                labels.append(new_label)
            break
    return texts, labels


texts, labels = load_sentirueval_2016('bank_train_2016.xml')
"""
print('Number of texts is {0}, number of labels is {1}.'.format(len(texts), len(labels)))

import random
for idx in random.choices(list(range(len(texts))), k=20):
    print('{0} => {1}'.format(labels[idx], texts[idx]))
"""
positive_tweets = [texts[idx] for idx in range(len(texts)) if labels[idx] == 'positive']
negative_tweets = [texts[idx] for idx in range(len(texts)) if labels[idx] == 'negative']
"""
for cur in positive_tweets[:5]:
    print(cur)
for cur in negative_tweets[:5]:
    print(cur)
"""

from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(lowercase=True, tokenizer=word_tokenize)
vectorizer.fit(texts)
# print(vectorizer.get_feature_names()[0:20])
# print(len(vectorizer.get_feature_names()))
print('\n')
X = vectorizer.transform(texts)
# print(type(X))  # vector
# print(texts[0])  # строка "ссылка взять кредит..."
# print(X[0])  # векторы каждого слова в формате (номер предложения, число) 1
# print(vectorizer.get_feature_names()[7773]) # это http

from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer().fit(X)
X_transformed = transformer.transform(X)
# print(X_transformed[0])  # выводит первое предложение и циферки тф
# высокий tf-idf - слово важно для понимания темы
# print(vectorizer.get_feature_names()[6318]) # высокий - ссылка с t.co
# print(vectorizer.get_feature_names()[7196]) # низкий - двоеточие

tokens_with_IDF = list(zip(vectorizer.get_feature_names(), transformer.idf_))
# print(tokens_with_IDF)  # тут лежат айтемы и их idf
"""
for feature, idf in tokens_with_IDF[0:20]:
    print('{0:.6f} => {1}'.format(idf, feature))
sorted_tokens_with_IDF = sorted(tokens_with_IDF, key=lambda it: (-it[1], it[0]))
for feature, idf in sorted_tokens_with_IDF[0:20]:
    print('{0:.6f} => {1}'.format(idf, feature))
"""

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2

selector = SelectPercentile(chi2, percentile=20)  # обозначение того, чем будем пользоваться
# print(X_transformed)  # выводит (номер предложения, номер слова) тф
# print(labels)  # +, -, 0
selector.fit(X_transformed, labels)
# print(selector.get_support(indices=True))  # индексы слов - ?
selected_tokens_with_IDF = [tokens_with_IDF[idx] for idx in selector.get_support(indices=True)]
"""
print(len(selected_tokens_with_IDF))
for feature, idf in selected_tokens_with_IDF[0:20]:
    print('{0:.6f} => {1}'.format(idf, feature))
"""
selected_and_sorted_tokens_with_IDF = sorted(selected_tokens_with_IDF, key=lambda it: (-it[1], it[0]))
for feature, idf in selected_and_sorted_tokens_with_IDF[0:100]:
    if feature.startswith('//t.co') or feature.isalpha() is False:
        pass
    else:
        print('{0:.6f} => {1}'.format(idf, feature))
