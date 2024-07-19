import tensorflow as tf
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import os

nltk.download('punkt')
nltk.download('wordnet')

class TechSupportChatbot:
    def __init__(self, max_sequence_length=20):
        self.lemmatizer = WordNetLemmatizer()
        self.intents = {}
        self.responses = {}
        self.all_words = []
        self.tags = []
        self.model = None
        self.max_sequence_length = max_sequence_length

    def load_dataset(self, data_dir):
        for intent in os.listdir(data_dir):
            intent_dir = os.path.join(data_dir, intent)
            if os.path.isdir(intent_dir):
                self.intents[intent] = []
                for file in os.listdir(intent_dir):
                    if file.endswith('.txt'):
                        with open(os.path.join(intent_dir, file), 'r', encoding='utf-8') as f:
                            content = f.read().splitlines()
                            self.intents[intent].extend(content[:-1])
                            self.responses[intent] = content[-1]

    def preprocess_text(self, text):
        tokens = nltk.word_tokenize(text.lower())
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def prepare_training_data(self):
        xy = []
        for intent, patterns in self.intents.items():
            self.tags.append(intent)
            for pattern in patterns:
                w = self.preprocess_text(pattern)
                self.all_words.extend(w)
                xy.append((w, intent))

        self.all_words = sorted(set(self.all_words))
        self.tags = sorted(set(self.tags))

        X_train = []
        y_train = []

        for (pattern_sentence, tag) in xy:
            bag = [self.all_words.index(word) + 1 if word in self.all_words else 0 for word in pattern_sentence]
            X_train.append(bag)
            label = self.tags.index(tag)
            y_train.append(label)

        X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=self.max_sequence_length)
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(self.tags))

        return np.array(X_train), np.array(y_train)

    def build_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(self.all_words) + 1, 128, input_length=self.max_sequence_length),
            tf.keras.layers.RNN(tf.keras.layers.SimpleRNNCell(64), return_sequences=False),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(self.tags), activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, epochs=300, batch_size=32):
        X_train, y_train = self.prepare_training_data()
        self.build_model()
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def save_model(self, model_path):
        self.model.save(model_path)

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict_intent(self, text):
        input_text = self.preprocess_text(text)
        input_data = [self.all_words.index(word) + 1 if word in self.all_words else 0 for word in input_text]
        input_data = tf.keras.preprocessing.sequence.pad_sequences([input_data], maxlen=self.max_sequence_length)
        
        predictions = self.model.predict(input_data)
        predicted_intent = self.tags[np.argmax(predictions)]
        
        return predicted_intent, self.responses.get(predicted_intent, "I'm not sure how to respond to that.")

def train_mode(data_dir, model_path):
    chatbot = TechSupportChatbot()
    chatbot.load_dataset(data_dir)
    chatbot.train()
    chatbot.save_model(model_path)
    print("Training completed and model saved.")

def operation_mode(data_dir, model_path):
    chatbot = TechSupportChatbot()
    chatbot.load_dataset(data_dir)
    chatbot.prepare_training_data() 
    chatbot.load_model(model_path)
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        intent, response = chatbot.predict_intent(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ML model used in iDTech chatbot project")
    parser.add_argument("mode", type=str, help="Operational mode to use: train | op")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing the dataset")
    parser.add_argument("--model_path", type=str, default="model.h5", help="Path to save/load the model")
    args = parser.parse_args()

    if args.mode == "train":
        train_mode(args.data_dir, args.model_path)
    elif args.mode == "op":
        operation_mode(args.data_dir, args.model_path)
    else:
        print("Invalid mode argument, please use 'train' or 'op'.")