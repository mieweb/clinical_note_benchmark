import os
import json
import pandas as pd
from glob import glob
from utils import parse_path

output_dir = './compare_output'
dir_keys = ['eval_model', 'cleaned']
json_keys = ['root_dir', 'model', 'prompt', 'idx']

if __name__ == '__main__':
    paths = glob(f'**/ai_eval.json', recursive=True)
    path_data = list(map(pd.read_json, paths))
    df = pd.concat(path_data)
    df.insert(0, 'full_path', paths)
    df.rename(columns={'model': 'eval_model'}, inplace=True)
    df = pd.merge(pd.json_normalize(df['full_path'].map(parse_path).values), df)
    df.drop('full_path', axis=1, inplace=True)
    df = df.melt(id_vars=[c for c in df.columns if c not in ['missing', 'added', 'text']], value_vars=['missing', 'added', 'text'])
    df.dropna(inplace=True)
    df = df[df['value'].str.len() > 0]
    df = df.explode('value')
    df['cleaned'] = df['cleaned'].map(lambda x: 'cleaned' if x else 'raw')

    for dirs, dir_groups in df.groupby(dir_keys):
        output = {}
        for group_name, group in dir_groups.groupby(['variable'] + json_keys):
            curr_dict = output
            for key in group_name:
                if key not in curr_dict.keys():
                    curr_dict[str(key)] = {}
                curr_dict = curr_dict[str(key)]
            if group_name[0] != 'text':
                group_data = pd.json_normalize(group['value'])
                group_data.set_index(group['timestamp'], inplace=True)
                clinical_concept = {k:v.to_list() for k,v in group_data.groupby('clinical_concept').groups.items()}
                if group_name[0] == 'missing':
                    evidence = {k:v.to_list() for k,v in group_data.groupby('evidence.snippet_A').groups.items()}
                else:
                    evidence = {k:v.to_list() for k,v in group_data.groupby('evidence.snippet_B').groups.items()}

                group_dict = {"clinical_concept": clinical_concept, "evidence": evidence}
            else:
                group_dict = list(zip(group['timestamp'].values, group['value'].values))
            curr_dict.update(group_dict)
        path = ('/').join(dirs)
        os.makedirs(f'{output_dir}/{path}', exist_ok=True)
        for key, value in output.items():
            with open(f'{output_dir}/{path}/{key}.json', 'w') as f:
                json.dump(value, f, indent=4)


