import os
import jieba

corpus = []
split_words = []
datapath = "./jyxstxtqj_downcc.com"
filelist = os.listdir(datapath)

stopwords_file_path = './/cn_punctuation.txt'
stopword_file = open(stopwords_file_path, "r", encoding='utf-8')
stop_words = stopword_file.read().split('\n')
stopword_file.close()

for filename in filelist:
    filepath = datapath + '/' + filename
    with open(filepath, "r", encoding="gb18030") as file:
        filecontext = file.read()
        filecontext = filecontext.replace(
            "本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com", '')
        filecontext = filecontext.replace("本书来自www.cr173.com免费txt小说下载站", '')
        filecontext = filecontext.replace('\n', '')  # 去除换行符
        filecontext = filecontext.replace(' ', '')  # 去除空格
        filecontext = filecontext.replace('\u3000', '')  # 去除全角空
        corpus.append(filecontext)
        file.close()

fw = open('corpus_sentence.txt', 'w', encoding='utf-8')
sentence = ''
for filecontext in corpus:
    for x in filecontext:
        if len(x.encode('utf-8')) == 3:  # and x not in stop_words:
            sentence += x
        if x in ['\n', '。', '？', '！', '；', '，'] and sentence != '\n':  # 以部分中文符号为分割换行
            # sentence += x
            for word in sentence:  # jieba.lcut(sentence):
                split_words.append(word)
            result = ' '.join(split_words)
            fw.write(result + ' ')  # 按行存入语料文件
            sentence = ''
            split_words = []
fw.close()
