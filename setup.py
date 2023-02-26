import os, json
from datasets import load_dataset



def preprocess_data(data_obj):
	preprocessed = []
	for elem in data_obj:
		text = elem['text'].replace('<br /><br />', ' ').lower()
		preprocessed.append({'text': text, 'label': elem['label']})
	return preprocessed



def save_data(data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-5000], data_obj[-5000:-2500], data_obj[-2500:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)        
        assert os.path.exists(f'data/{key}.json')



def main():
	orig = load_dataset('imdb')
	orig = orig['train'] + orig['test']
	processed = preprocess_data
	save_data(processed)



if __name__ == '__main__':
	main()