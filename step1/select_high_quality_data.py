import json
import argparse

quality_threshold=85
input_path='../data/qwen1_5_7b/qwen1_5_7b_rated_result.jsonl'
output_path='../data/qwen1_5_7b/qwen1_5_7b_high_quality_data.json'

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str,default=input_path)
    parser.add_argument("--output_path", type=str,default=output_path)
    parser.add_argument("--quality_threshold", type=int, default=quality_threshold)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    high_quality_data=[]
    with open(args.input_path, 'r',encoding='utf-8') as f:
        for line in f:
            data=json.loads(line)
            if data['score']>=args.quality_threshold:
                high_quality_data.append(data)

    with open(args.output_path,'w',encoding='utf-8') as f:
        json.dump(high_quality_data, f, ensure_ascii=False, indent=4)