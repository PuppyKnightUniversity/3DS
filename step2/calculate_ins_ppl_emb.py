import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch.nn as nn
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_length=1024

model_name_or_path = "../../llm/hub/Qwen/Qwen1.5-7B-Chat"

input_path="../data/qwen1_5_7b/qwen1_5_7b_high_quality_data.json"
output_path="../data/qwen1_5_7b/meddata_qwen1_5_7b_ins_emb_ppl.pt"

model_name="qwen"


# instruction model generation template
baichuan2_template="""<s><reserved_106>{question}<reserved_107>"""

qwen_template="""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant\n"""


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path",type=str, default=model_name_or_path)
    parser.add_argument("--input_path", type=str,default=input_path)
    parser.add_argument("--output_path", type=str,default=output_path)
    parser.add_argument('--model_name',type=str,default=model_name)
    parser.add_argument('--max_length', type=int, default=max_length)

    return parser.parse_args()


def text_perplexity_and_embedding(tokenizer, model, text, max_length):

    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    if input_ids.dtype != torch.long:
        try:
            input_ids = input_ids.long()
        except RuntimeError as e:
            print(f"Error converting input_ids to LongTensor: {e}")
            return None, None

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids.contiguous(),output_hidden_states=True)
    loss = outputs.loss
    perplexity = torch.exp(loss)

    hidden_states = outputs.hidden_states
    embeddings = hidden_states[-1]
    sentence_embedding = embeddings.mean(dim=1)

    return perplexity.to('cpu'), sentence_embedding.to('cpu')

def main():
    args = parse_arguments()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, output_hidden_states=True, trust_remote_code=True,
                                                 torch_dtype=torch.float16, device_map='auto')
    model.to(device)
    model.eval()

    continue_id=set()
    if os.path.exists(args.output_path):
        pre_result=torch.load(args.output_path,map_location=torch.device('cpu'))
        for data in pre_result:
            continue_id.add(data['id'])
    else:
        pre_result=[]

    datas=[]
    with open(args.input_path,'r',encoding='utf-8') as f:
        tmp_datas=json.load(f)
        for data in tmp_datas:
            if data['id'] in continue_id:
                continue
            else:
                datas.append(data)


    new_datas=[]
    cnt=0
    for i in tqdm(range(len(datas))):
        data=datas[i]
        new_data={
            'id':data['id']
        }

        question = data['instruction']
        if args.model_name.startswith("baichuan"):
            prompt=baichuan2_template.replace("{question}", question)
        else:
            prompt=qwen_template
            prompt=prompt.replace("{question}",question)

        if i==0:
            print(prompt)

        ppl_ins,emb_ins=text_perplexity_and_embedding(tokenizer,model,prompt,args.max_length)

        if ppl_ins is None or emb_ins is None:
            continue
        else:
            new_data['ppl_ins']=[ppl_ins]
            new_data['emb_ins']=[emb_ins]

            new_datas.append(new_data)
            cnt += 1
            if cnt % 1000 == 0:
                torch.save(new_datas + pre_result, args.output_path)
    torch.save(new_datas,args.output_path)

if __name__ == '__main__':
    main()