import pandas as pd
import os


def extract_info(file_name):
    df = pd.read_excel(file_name)

    return df


def combine_results(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    df = pd.DataFrame()
    for file in files:
        if not df.empty:
            df = pd.concat([df, pd.read_excel(file)])
        else:
            df = pd.read_excel(file)
    return df.reset_index(drop=True)


def get_unique_papers(df1, df2):
    result = df1[-df1['Article Title'].isin(df2['Article Title'])]
    return result


def extract_only_letters(word):
    result = ""
    for c in word:
        if c.isalpha():
            result += c
    return result


def clean_word(word):
    clean_from_chars = ',;.\'"!?*'
    for c in clean_from_chars:
        word = word.replace(c, '')
    return word


def extract_dataset_names(df):
    def get_datasets(abstract):
        blacklist = ['eeg', 'audiovisual', 'ser', 'fmri', 'cnn', 'svm', 'bovw',
                     'auc', 'hci', 'hog', 'swift', 'uk', 'usa', 'microexpression', 'ii',
                     'hrv', 'ecg', 'hmm', 'iot', 'youtube', 'imdb', 'softmax', 'vgg',
                     'ai', 'ml', 'dnn', 'dcnn', 'pca', 'lda', 'sift', 'mri', 'mfcc',
                     'cnns', 'pubmed', 'rgb', 'mlp', 'covid', 'methods', 'results', 'resnet',
                     'roi', 'knn', 'unet', 'shortterm', 'background', 'matlab', 'tv', 'iou', 'cca',
                     'bow', 'rnn', 'gru', 'fps', 'eegbased', 'densenet', 'svms', 'objective',
                     'db', 'ann', 'materials', 'melfrequency', 'lidar', 'mrmri', 'knearest',
                     'violajones', 'fscore', 'inceptionv', 'ssd', 'gan', 'bv', 'cnnbased', 'yolo',
                     'ct', 'lstm', 'conclusion', 'rgbd', 'alexnet', 'hmms', 'mfccs', 'conclusions',
                     'photooptical', 'adaboost', 'dl', 'lbp', 'dnns', 'ieee', 'dr', 'rcnn',
                     'rmse', 'roc', 'nlp', 'yolov', 'elm', 'cad', 'ci', 'ad', 'pd', 'au',
                     'modis', 'dbn', 'rois', 'spie', 'rf', 'ir', 'wm', 'md', 'asd', 'map',
                     'ga', 'fau', 'pet', 'cc', 'snr', 'ar', 'cd', 'lr', 'dti', 'gmm', 'ict',
                     'fp', 'eer', 'mes', 'me', 'er', 'si', 'hr', 'nir', 'surf', 'glcm',
                     'sar', 'asr', 'mr', 'aus', 'fa', 'tbi', 'ms', 'amd', 'dwt', 'mci',
                     'sd', 'oct', 'dct', 'us', 'uav', 'tm', ]
        datasets = ''
        for word in str(abstract).split():
            cap = [c for c in word if c.isupper()]
            if len(cap) > 1 and extract_only_letters(word.lower()) not in blacklist:
                datasets += f'{word} '
        datasets = ', '.join(list(set(datasets.split())))
        return datasets

    df['Datasets'] = df['Abstract'].apply(get_datasets)
    return df


def extract_datasets_popularity_rate(df, abbreviations):
    datasets = df['Datasets']
    all_datasets = ''
    for dataset in datasets:
        all_datasets += ' '.join([extract_only_letters(d).lower() for d in dataset.split(', ')]) + ' '
    all_datasets = all_datasets.split()

    from collections import Counter
    result = Counter(all_datasets)

    result_df = pd.DataFrame(result.values(), index=result.keys(), columns=['Occurrence'])
    result_df = result_df.sort_values(by='Occurrence', ascending=False)

    result_df['Abbreviation'] = result_df.index.map(lambda x: abbreviations[x] if x in abbreviations else '')

    return result_df


def save_df(df, output_path):
    df.to_excel(output_path)


def extract_abbreviation(main_df: pd.DataFrame):
    keywords_dict = {}

    for idx, row in main_df.iterrows():
        keywords = row['Datasets'].split(', ')
        for kw in keywords:
            if kw != '' and kw[0] == '(' and kw[-1] == ')' and extract_only_letters(kw) not in keywords_dict:
                abstract = str(row['Abstract'])
                kw_index = abstract.find(kw[:5])
                assert kw_index > 0, f'Some dataset has non letter chars: {kw} from: {abstract}'
                sentence_start_idx = abstract.rfind('.', 0, kw_index)
                sentence_end_idx = abstract.find('.', kw_index)
                sentence = abstract[sentence_start_idx + 1:sentence_end_idx]
                keywords_dict[extract_only_letters(kw).lower()] = sentence
    return keywords_dict


if __name__ == '__main__':
    file_path = "./files/combined_dataset_papers_wos_pubmed.xls"
    new_df = pd.read_excel(file_path)
    new_df = extract_dataset_names(new_df)
    save_df(new_df, f'{file_path[:-4]}_extended.xls')

    abbreviations = extract_abbreviation(new_df)
    popularity_df = extract_datasets_popularity_rate(new_df, abbreviations)
    save_df(popularity_df, f'{file_path[:-4]}_pop.xls')
    print('Done')
