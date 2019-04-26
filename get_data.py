from json import dumps
from collections import defaultdict
import praw, configparser
#from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
#lemmatizer.lemmatize("Dont")
config = configparser.ConfigParser()
config.read('oauth.conf')
client_key = config['keys']['client_id']
secret_key = config['keys']['client_secret']
reddit = praw.Reddit(client_id=client_key, client_secret=secret_key, user_agent='python nlp project by /u/viraj25j')
#data_file = open("data.json", "w")


def clean_data(text):
    text = sub(r"what's", "what is ", text)
    text = sub(r"\'s", " ", text)
    text = sub(r"\'ve", " have ", text)
    text = sub(r"can't", "cannot ", text)
    text = sub(r"n't", " not ", text)
    text = sub(r"i'm", "i am ", text)
    text = sub(r"\'re", " are ", text)
    text = sub(r"\'d", " would ", text)
    text = sub(r"\'ll", " will ", text)
    text = sub(r",", " ", text)
    text = sub(r"\.", " ", text)
    text = sub(r"!", " ! ", text)
    text = sub(r"\/", " ", text)
    text = sub(r"\^", " ^ ", text)
    text = sub(r"\+", " + ", text)
    text = sub(r"\-", " - ", text)
    text = sub(r"\=", " = ", text)
    return(text)

text = defaultdict(list)
for post in reddit.subreddit('ucf').top(limit=2):
    post.comments.replace_more(limit=32, threshold=0)
    title = [x for x in clean_data(post.title) if (x.isalnum() or x == " ")]
    for comment in post.comments.list():
        #p_comm = comment.body.lower())
        #p_comm = [lemmatizer.lemmatize(word) for word in comment.body.lower()]
        text["".join([y for y in post.title if (y.isalnum() or y == " ")])].append("".join([x for x in comment.body.lower() if (x.isalnum() or x == " ")]))
print(text)
           
#data_file.write(dumps(text))
#data_file.close()
