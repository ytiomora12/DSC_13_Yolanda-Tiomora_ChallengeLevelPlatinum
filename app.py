# PACKAGE
import pandas as pd
import re
import numpy as np

import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from flask import request, Flask, jsonify
from flasgger import Swagger, LazyString, LazyJSONEncoder, swag_from

# DEFINING TEXT CLEANSING
def text_cleansing(text):
    text = text.lower() #lowercase
    text = text.strip() #menghapus spasi di awal dan akhir
    text = re.sub(r'\buser\b|\brt\b|\bamp\b|(\bx[\da-f]{2})', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\n', ' ', text, flags=re.IGNORECASE)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text)
    text = re.sub(r'(.)\1\1+', r'\1', text) #menghapus karakter berulang
    text = re.sub('[^0-9a-zA-Z]+', ' ', text) #menghapus karakter non-alpanumerik
    text = re.sub(r'[øùºðµ¹ª³]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'â', 'a', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip() #menghapus spasi berlebih dan mengganti dengan satu spasi
    text = re.sub(r'^\s+$', '', text) #menghapus seluruh kalimat yg hanya berisi spasi
    return text

# df_kamus AS A DICT
df_kamus = pd.read_csv("utils/new_kamusalay.csv", encoding="latin-1", header=None)
df_kamus_map = dict(zip(df_kamus[0], df_kamus[1]))
def normalize(text):
    return ' '.join([df_kamus_map[word] if word in df_kamus_map else word for word in text.split(' ')])

# APPLY TEXT CLEANSING AND DICT
def preprocess_apply(text):
    text = text_cleansing(text)
    text = normalize(text)
    return text

# NN
## Load result of Feature Extraction process from NN
file = open('resources_of_nn/feature.p','rb')
vect = pickle.load(file)
file.close()

## Load model from NN
nn_model = pickle.load(open('resources_of_nn/model.p', 'rb'))

# DEFINE FEATURE EXTRACTION PARAMETER AND TOKENIZER CLASS
with open ('utils/total_data', 'rb') as fp:
    total_data = pickle.load(fp)

max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split= ' ', lower=True)

# DEFINE LABEL SENTIMENT
sentiment = ['negative', 'neutral', 'positive']

# LSTM
## Load result of Feature Extraction process from LSTM
file = open('resources_of_lstm/x_pad_sequences.pickle','rb')
lstm_feature = pickle.load(file)
file.close()

## Load model from LSTM
lstm_model = load_model('resources_of_lstm/model.h5')

# SWAGGER UI
app = Flask(__name__)

app.json_encoder = LazyJSONEncoder
swagger_template = dict(
info = {
    'title': LazyString(lambda: 'API Documentation for Machine Learning and Deep Learning'),
    'version': LazyString(lambda: '1.0.0'),
    'description': LazyString(lambda: 'Dokumentasi API untuk Machine Learning dan Deep Learning')
    },
    host = LazyString(lambda: request.host)
)

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json'
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}
swagger = Swagger(app, template=swagger_template,
                 config=swagger_config)


# Define endpoint for Sentiment Analysis using NN
@swag_from(r"C:\Users\ghifa\CoDe\Platinum_Chal\docs\nn.yml", methods=['POST'])
@app.route('/nn', methods=['POST'])
def cnn():
    # Get text
    original_text = request.form.get('text')
    # Cleansing
    text = preprocess_apply(original_text)
    # Feature extraction
    text_feature = vect.transform([text])
    # Inference
    get_sentiment = nn_model.predict(text_feature)[0]

#     OUTPUT JSON RESPONSE
    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis using CNN",
        'data': {
            'text': original_text,
            'sentiment': get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data


# Define endpoint for Sentiment Analysis using NN from file
@swag_from(r"C:\Users\ghifa\CoDe\Platinum_Chal\docs\nn_file.yml", methods=['POST'])
@app.route('/nn-file', methods=['POST'])
def nn_file():

    # Upladed file
    file = request.files.getlist('file')[0]
    # Import file csv ke Pandas
    df = pd.read_csv(file, encoding='latin-1')
    # Get text from file in "List" format
    texts = df.Tweet.to_list()
    
    # Loop list or original text and predict to model
    text_with_sentiment = []
    for original_text in texts:
        # Cleansing
        text = [preprocess_apply(original_text)]
        # Feature extraction
        text_feature = vect.transform(text)
        # Inference
        get_sentiment = nn_model.predict(text_feature)[0]
        
        # Predict "text_clean" to the Model. And insert to list "text_with_sentiment".
        text_with_sentiment.append({
            'text': original_text,
            'sentiment': get_sentiment
        })
    
#     OUTPUT JSON RESPONSE
    json_response = {
        'status_code': 200,
        'description': "Teks yang sudah diproses",
        'data': text_with_sentiment,
    }
    response_data = jsonify(json_response)
    return response_data

# Define endpoint for Sentiment Analysis using LSTM
@swag_from(r"C:\Users\ghifa\CoDe\Platinum_Chal\docs\lstm.yml", methods=['POST'])
@app.route('/lstm', methods=['POST'])
def lstm():
    # Cleansing
    original_text = request.form.get('text')
    text = [preprocess_apply(original_text)]
    # Feature extraction
    tokenizer.fit_on_texts(total_data)
    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=lstm_feature.shape[1])
    # Inference
    prediction = lstm_model(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

#     OUTPUT JSON RESPONSE
    json_response = {
        'status_code': 200,
        'description': "Hasil Teks dan Sentimennya",
        'data': {
            'text': original_text,
            'sentiment': get_sentiment
        }
    }
    
    response_data = jsonify(json_response)
    return response_data

# Define endpoint for Sentiment Analysis using LSTM from file
@swag_from(r"C:\Users\ghifa\CoDe\Platinum_Chal\docs\lstm_file.yml", methods=['POST'])
@app.route('/lstm-file', methods=['POST'])
def lstm_file():
    
#     Upladed file and import
    filein = request.files.getlist('filein')[0]
    df = pd.read_csv(filein,encoding="latin-1")
#     pd.set_option('display.max_colwidth', None)

#     Get text from file in "List" format
    texts = df.Tweet.to_list()
    
    tokenizer.fit_on_texts(total_data)

#     Loop list or original text and predict to model
    text_with_sentiment = []
    for original_text in texts:
        # Cleansing
        text = [preprocess_apply(original_text)]
        # Feature extraction
        feature = tokenizer.texts_to_sequences(text)
        feature = pad_sequences(feature, maxlen=lstm_feature.shape[1])
        # Inference
        prediction = lstm_model.predict(feature)
        get_sentiment = sentiment[np.argmax(prediction[0])]

        # Predict "text_clean" to the Model. And insert to list "text_with_sentiment".
        text_with_sentiment.append({
            'text': original_text,
            'sentiment': get_sentiment
        })
    
#     OUTPUT JSON RESPONSE
    json_response = {
        'status_code': 200,
        'description': "File yang sudah diproses",
        'data': text_with_sentiment
    }
    
    response_data = jsonify(json_response)
    return response_data

if __name__ == '__main__':
    app.run()