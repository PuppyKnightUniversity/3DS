import os
import requests
import json
from tqdm import tqdm
from multiprocessing import Pool, Manager
import argparse


class InputLengthError(requests.RequestException):
    """The length of input exceeds the max length"""

class InvalidKeyError(requests .RequestException):
    """The key is invalid."""

input_path='../data/qwen1_5_7b/qwen1_5_7b_high_quality_data.json'
output_path='../data/qwen1_5_7b/qwen1_5_7b_high_quality_generated.jsonl'


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
        "model": "llama2",
        "messages": [
            {
                "role":"system",
                "content":""
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "do_sample":False,
        "num_return_sequences": 1
    })

    try:
        raw_response = requests.post('http://0.0.0.0:8001/v1/chat/completions', headers=headers, data=data)
        raw_response.raise_for_status()  # Raises stored HTTPError, if one occurred.

        res = json.loads(raw_response.text)
        ans = res['choices'][0]['message']['content']
        return ans
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

    prompt=data['instruction']

    ans=call_model(prompt)
    if ans.startswith("error"):
        print(ans)
        return False
    result = data
    result['generated_output'] = ans
    with lock:
        with open(out_path, 'a', encoding='utf-8') as outfile:
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
    return True


if __name__ == "__main__":
    args=parse_arguments()

    continue_id=set()
    if os.path.exists(args.output_path):
        with open(args.output_path, "r+", encoding='utf-8') as f:
            out_data = f.readlines()
            for line in out_data:
                line = json.loads(line)
                continue_id.add(line['id'])

    datas = []
    cnt = 0
    with open(args.input_path, 'r', encoding='utf-8') as f:
        data=json.load(f)
        for item in data:
            if item['id'] in continue_id:
                continue
            datas.append(item)

    manager = Manager()
    lock = manager.Lock()
    with Pool(50) as p:
        for example_res in tqdm(p.imap_unordered(worker, [(data, args.output_path, lock) for data in datas]),
                                total=len(datas)):
            if example_res != True:
                print(example_res)
                with open('log.txt', 'a') as f:
                    f.write(str(example_res) + '\n')
