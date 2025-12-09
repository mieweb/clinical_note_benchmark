from dotenv import load_dotenv
from glob import glob
import os
import json
import pandas as pd
from tqdm import tqdm
import re    
import requester
from rouge import Rouge



def get_standard_path(idx, standard_dir='standards'):
    if f'{str(idx)}.txt' in os.listdir(standard_dir):
        path = os.readlink(os.path.join(standard_dir, f'{str(idx)}.txt'))
        return path
    else:
        raise FileNotFoundError(f'Standard note for idx {str(idx)} not found in {standard_dir}')

def arg_to_regex(arg, all_pattern):
    if arg == 'all':
        return all_pattern
    elif isinstance(arg, list):
        return ('|').join(map(str,arg))
    else:
        return str(arg)
    
def search_file_paths(filename='gen_note.txt', results_dir_path='results', idxs='all', models='all', prompts='all'):
    idxs = arg_to_regex(idxs, r'\d+')
    models = arg_to_regex(models, r'.+')
    prompts = arg_to_regex(prompts, r'g\d+')
    pattern = re.compile(rf'{results_dir_path}/({idxs})/({models})/({prompts})/(\d+\.\d+)/{filename}')
    glob_paths = glob(f'{results_dir_path}/**/{filename}', recursive=True)
    filtered_paths = list(filter(pattern.match, glob_paths))
    return filtered_paths


def parse_path(path, include_full_path=True):
    pattern = re.compile(rf'(.*)/(\d+)/(.+)/(.+)/(\d+\.\d+)/(.+)')
    match = pattern.match(path)
    assert match
    if include_full_path:
        return {
            'full_path': path,
            'root_dir': match[1],
            'idx': int(match[2]),
            'model': match[3],
            'prompt': match[4],
            'timestamp': match[5]
        }
    else:
        return {
            'root_dir': match[1],
            'idx': int(match[2]),
            'model': match[3],
            'prompt': match[4],
            'timestamp': match[5]
        }

def read(path):
    with open(path, 'r') as file:
        content = file.read()
    return content


def get_metadata_df(paths):
    notes = list(map(read, paths))
    metadata = list(map(parse_path, paths))
    df = pd.DataFrame.from_records(metadata)
    df['note'] = notes
    return df


def write_reports(path, df, overwrite=False):
    if os.path.exists(path) and not overwrite:
        existing_report = pd.read_json(path)
        pd.concat((existing_report, df)).reset_index(drop=True).to_json(path, orient='records', indent=4)
    else:
        df.to_json(path, orient='records', indent=4)


def melt_rouge_scores(df):
    rouge_types = [col for col in df.columns if re.match(r'rouge-*', col)]
    other_cols = [col for col in df.columns if col not in rouge_types]
    melted = df.melt(id_vars=other_cols, value_vars=rouge_types)
    melted = pd.concat((melted,pd.json_normalize(melted['value'])), axis=1)
    melted.rename(columns={'variable': 'rouge_type'}, inplace=True)
    melted.drop(columns=['value'], inplace=True)
    other_cols.append('rouge_type')
    melted = melted.melt(id_vars=other_cols, value_vars=['r', 'p', 'f'])
    melted.rename(columns={'variable': 'metric'}, inplace=True)
    return melted

def pivot_df(melted_df, pivot_index, aggfunc=None):
    if aggfunc:
        melted_df = melted_df.pivot_table(index=pivot_index, columns=['metric'], values=['value'], aggfunc=aggfunc)
    else:
        melted_df = melted_df.pivot(index=pivot_index, columns=['metric'], values=['value'])
    melted_df.columns = melted_df.columns.get_level_values(1)
    melted_df.columns.names = [None]
    return melted_df


def get_rouge_report(path):
    path_data = parse_path(path, False)
    df = pd.read_json(path).melt(id_vars=["standard_note_path", "cleaned"], value_vars=['r', 'p', 'f'], var_name='metric_type', value_name='score')
    rouge_type = path.split('/')[-1].split('.')[0]
    df.insert(2, 'rouge_type', rouge_type)
    df = df.assign(**path_data)
    df = df.loc[:, list(path_data.keys()) + list(df.columns[:-len(path_data)])] # reorder columns
    return df

def get_most_recent_timestamp(root_dir, model, prompt, idx):
    path = os.path.join(root_dir, str(idx), model, prompt)
    timestamps = os.listdir(path)
    timestamps = sorted(timestamps, key=lambda x: float(x), reverse=True)
    return timestamps[0]

