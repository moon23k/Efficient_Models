import os, json
from datasets import load_dataset



def preprocess_data(data_obj):
	preprocessed = []
	for elem in data_obj:
		preprocessed.append({'text': elem['text'].replace('<br />', '').lower(), 
                             'label': elem['label']})
	return preprocessed



def save_data(data_obj):
    #split data into train/valid/test sets
    train, valid = data_obj[:-5000], data_obj[-5000:]
    data_dict = {k:v for k, v in zip(['train', 'valid'], [train, valid])}

    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)        
        assert os.path.exists(f'data/{key}.json')



def main():
	orig = load_dataset('imdb')
	processed = preprocess_data(orig['train']) + preprocess_data(orig['test'])
	save_data(processed)



if __name__ == '__main__':
	main()