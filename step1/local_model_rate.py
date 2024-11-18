import os, json
#import openai
import requests
import json
from tqdm import tqdm
from multiprocessing import Pool, Value, Lock, Manager
from rating_prompt import rating_prompt
import random
import argparse

class InputLengthError(requests.RequestException):
    """The length of input exceeds the max length"""


class InvalidKeyError(requests .RequestException):
    """The key is invalid."""

rating_prompt=rating_prompt

input_path='../data/meddata_singleround_data.json'
output_path= '../data/qwen1_5_7b/qwen1_5_7b_rated_result.jsonl'

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str,default=input_path)
    parser.add_argument("--output_path", type=str,default=output_path)

    return parser.parse_args()



def call_model(prompt):
    headers = {
        "Content-Type": "application/json"
    }
    data = json.dumps({
        "model": "Qwen1___5-7B",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })

    try:
        raw_response = requests.post('http://0.0.0.0:8001/v1/chat/completions', headers=headers, data=data)
        raw_response.raise_for_status()  # Raises stored HTTPError, if one occurred.

        res = json.loads(raw_response.text)
        words = res['choices'][0]['message']['content']
        #print(words)
        return float(words)
    except requests.HTTPError as e:
        if e.response.status_code in (400, 401):
            # Handle specific error codes if needed
            error_info = e.response.json()
            error_message = error_info.get('error', {}).get('message', 'Unknown error')
            return f"error: {error_message}"
        else:
            return "error: Unexpected error"
    except Exception as e:
        return f"error: {str(e)}"



def worker(args):
    data,out_path, lock = args

    qa_pairs="user: "+data['instruction']+'\nassistant: '+data['output']

    prompt=rating_prompt.replace('<qa_pairs>',qa_pairs)

    ans=call_model(prompt)
    if isinstance(ans,str) and ans.startswith("error"):
        print(ans)
        return False
    result = data
    result['score']=ans
    with lock:
        with open(out_path, 'a', encoding='utf-8') as outfile:
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
    return True


if __name__ == "__main__":
    args = parse_arguments()

    continue_id=set()
    if os.path.exists(args.output_path):
        with open(args.output_path, "r+", encoding='utf-8') as f:
            out_data = f.readlines()
            for line in out_data:
                line = json.loads(line)
                continue_id.add(line['id'])

    with open(args.input_path, 'r', encoding='utf-8') as f:
       tmp_datas = json.load(f)
    datas=[]
    for data in tmp_datas:
        if data['id'] not in continue_id:
            datas.append(data)

    manager = Manager()
    lock = manager.Lock()
    with Pool(50) as p:
        for example_res in tqdm(p.imap_unordered(worker, [(data, out_path, lock) for data in datas]),
                                total=len(datas)):
            if example_res != True:
                print(example_res)
                with open('log.txt', 'a') as f:
                    f.write(str(example_res) + '\n')