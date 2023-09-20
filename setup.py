import os, re, json, argparse
from datasets import load_dataset




def fetch_imdb(orig_data):
    fetched = []
    tot_volumn = 12000
    class_volumn = 12000 // 2
    neg_cnt, pos_cnt = 0, 0
    neg_data, pos_data = [], []

    train_data = [x for x in orig_data['train']]
    test_data = [x for x in orig_data['test']]
    concat_data = train_data + test_data

    for elem in concat_data:
        if neg_cnt + pos_cnt == tot_volumn:
            break
        text = elem['text'].replace('<br />', '').lower()
        label = elem['label']

        if label == 0 and pos_cnt < class_volumn:
            pos_cnt += 1
            neg_data.append({'x': text, 'y': label})

        elif label == 1 and neg_cnt < class_volumn:
            neg_cnt += 1
            pos_data.append({'x': text, 'y': label})

    for neg_elem, pos_elem in zip(neg_data, pos_data):
        fetched.append(neg_elem)
        fetched.append(pos_elem)

    return fetched



def fetch_agnews(orig_data):
    fetched = []
    tot_volumn = 12000
    class_volumn = 12000 // 4
    class1_cnt, class2_cnt, class3_cnt, class4_cnt = 0, 0, 0, 0
    class1_data, class2_data, class3_data, class4_data = [], [], [], []

    for elem in orig_data['train']:
        curr_volumn = class1_cnt + class2_cnt + class3_cnt + class4_cnt
        
        if curr_volumn == tot_volumn:
            break

        text = elem['text'].lower()
        text = re.sub(r'\\+', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'&lt;b&gt;', ' ', text)
        label = elem['label']

        if label == 0 and class1_cnt < class_volumn:
            class1_cnt += 1
            class1_data.append({'x': text, 'y': label})
        
        elif label == 1 and class2_cnt < class_volumn:
            class2_cnt += 1
            class2_data.append({'x': text, 'y': label})
        
        elif label == 2 and class3_cnt < class_volumn:
            class3_cnt += 1
            class3_data.append({'x': text, 'y': label})
        
        elif label == 3 and class4_cnt < class_volumn:
            class4_cnt += 1
            class4_data.append({'x': text, 'y': label})                        

    for elem1, elem2, elem3, elem4 in zip(class1_data, class2_data, class3_data, class4_data):
        fetched.append(elem1)
        fetched.append(elem2)
        fetched.append(elem3)
        fetched.append(elem4)

    return fetched




def save_data(task, data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-2000], data_obj[-2000:-1000], data_obj[-1000:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{task}/{key}.json', 'w') as f:
            json.dump(val, f)
        assert os.path.exists(f'data/{task}/{key}.json')



def main(task):
    os.makedirs(f'data/{task}', exist_ok=True)

    orig_data = load_dataset(task)

    fetched_data = fetch_imdb(orig_data) \
                    if task == 'imdb' \
                    else fetch_agnews(orig_data)

    save_data(task, fetched_data)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', required=True)

    args = parser.parse_args()
    assert args.task.lower() in ['ag_news', 'imdb'] 
    main(args.task)