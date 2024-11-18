import numpy as np
import torch
import json
from tqdm import tqdm
import spacy
from kcenter import *
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os

max_length=1024
sample_num=5000
low_th=25
up_th=75

model_name_or_path="../../llm/hub/Qwen/Qwen1.5-7B-Chat"
emb_pt_path='../data/qwen1_5_7b/meddata_qwen1_5_7b_ins_emb_ppl.pt'
atten_pt_path='../data/qwen1_5_7b/meddata_qwen1_5_7b_atten.pt'
json_data_path='../data/qwen1_5_7b/qwen1_5_7b_high_quality_data.json'
jsonl_data_path='../data/qwen1_5_7b/qwen1_5_7b_high_quality_generated.jsonl'
analysis_save_path='../data/qwen1_5_7b/meddata_qwen1_5_7b_analysis_arrays.pt'
json_save_path='../data/qwen1_5_7b/meddata_qwen1_5_7b_25_75_5k.json'

atten_method='mean'
model_name='qwen'

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, help="model to be fine-tuned", default=model_name_or_path)
    parser.add_argument("--json_data_path", type=str, help="original data to be selected", default=json_data_path)
    parser.add_argument("--emb_pt_path", type=str, help="instruction embedding and ppl calculated by calculate_emb",
                        default=emb_pt_path)
    parser.add_argument("--atten_pt_path", type=str,
                        help="attention, ppl, and token loss for responses A and A' calculated by calculate_attention",
                        default=atten_pt_path)
    parser.add_argument("--jsonl_data_path", type=str, help="model generated answers", default=jsonl_data_path)
    parser.add_argument("--analysis_save_path", type=str,
                        help="calculated emb/ppl/weighted loss arrays, ready for selection", default=analysis_save_path)
    parser.add_argument('--json_save_path', type=str, help="storage path for selected data", default=json_save_path)

    parser.add_argument('--atten_method', type=str, help="attention aggregation method, mean/max", default=atten_method)

    parser.add_argument('--sample_num', type=int, help="number of selected samples", default=sample_num)
    parser.add_argument('--low_th', type=int, help="lower threshold for difficulty filtering", default=low_th)
    parser.add_argument('--up_th', type=int, help="upper threshold for difficulty filtering", default=up_th)
    parser.add_argument('--max_length', type=int, help="maximum sequence length allowed", default=max_length)
    parser.add_argument('--model_name', type=str, help="name of the model", default=model_name)

    return parser.parse_args()

def weighted_loss(tokenizer,nlp,text,target_span,loss_list,atten_list,max_length):
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to('cpu')
    start_index = text.rfind(target_span)
    text_temp = text[:start_index]
    token_id_temp = tokenizer.encode(text_temp)
    start_token = len(token_id_temp)
    end_token_real = input_ids.shape[1]

    loss_list = loss_list[start_token - 1:end_token_real - 1]

    atten_list=atten_list[start_token:]
    doc = nlp(target_span)
    stop_words = set(
        token.text for token in doc if token.pos_ in ['PART', 'DET', 'PUNCT', 'CCONJ', 'ADP', 'CONJ', 'SCONJ', 'INTJ'])

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])  # [start_token:])
    tokens = [tokenizer.convert_tokens_to_string([token]) for token in tokens]
    tokens=tokens[start_token:]
    assert len(loss_list)==len(tokens)
    assert len(atten_list)==len(tokens)

    filtered_loss = []
    filtered_attention = []

    for token, loss, attention in zip(tokens, loss_list, atten_list):
        if token not in stop_words:
            filtered_loss.append(loss)
            filtered_attention.append(attention)

    total_attention = sum(filtered_attention)
    normalized_attention = [att / total_attention for att in filtered_attention]
    weighted_loss = sum(l * a for l, a in zip(filtered_loss, normalized_attention))

    return weighted_loss


baichuan2_template="""<s><reserved_106>{question}<reserved_107>"""

qwen_template="""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant\n"""


def main():
    args = parse_arguments()

    # if analysis file already exits, select data directly
    if os.path.exists(args.analysis_save_path):
        analysis_results=torch.load(args.analysis_save_path,map_location=torch.device('cpu'))
        emb_vectors = analysis_results['emb_vectors']
        mean_weighted_loss_A_array = analysis_results['mean_weighted_loss_A_array']
        mean_weighted_loss_A1_array = analysis_results['mean_weighted_loss_A1_array']
        max_weighted_loss_A_array = analysis_results['max_weighted_loss_A_array']
        max_weighted_loss_A1_array = analysis_results['max_weighted_loss_A1_array']
        ppl_ins_array = analysis_results['ppl_ins_array']
        final_json_data=analysis_results['final_json_data']
    else:
        nlp=spacy.load('zh_core_web_sm')

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,trust_remote_code=True)

        with open(args.json_data_path, "r") as f:
            tmp_json_data = json.load(f)

        emb_pt_data = torch.load(args.emb_pt_path, map_location=torch.device('cpu'))
        atten_pt_data = torch.load(args.atten_pt_path,map_location=torch.device('cpu'))

        generated_output_dict={}
        with open(args.jsonl_data_path,"r",encoding='utf-8') as f:
            for line in f:
                line=json.loads(line)
                generated_output_dict[line['id']]=line['generated_output']

        loss_dict = {}
        mean_atten_dict={}
        max_atten_dict={}
        for i in tqdm(range(len(atten_pt_data))):
            loss_dict[atten_pt_data[i]['id']] = [atten_pt_data[i]['loss'][0],atten_pt_data[i]['loss'][1]]
            max_atten_dict[atten_pt_data[i]['id']]=[atten_pt_data[i]['max_token_atten'][0],atten_pt_data[i]['max_token_atten'][1]]
            mean_atten_dict[atten_pt_data[i]['id']] = [atten_pt_data[i]['mean_token_atten'][0],atten_pt_data[i]['mean_token_atten'][1]]

        ppl_ins_dict={}
        emb_dict={}
        for i in tqdm(range(len(emb_pt_data))):
            ppl_ins_dict[emb_pt_data[i]['id']]=emb_pt_data[i]['ppl_ins']
            emb_dict[emb_pt_data[i]['id']]=emb_pt_data[i]['emb_ins']

        emb_list = []
        ppl_ins_list=[]
        mean_weighted_loss_A_list = []
        mean_weighted_loss_A1_list = []
        max_weighted_loss_A_list = []
        max_weighted_loss_A1_list = []
        final_json_data=[]

        json_data=[]
        for i in tqdm(range(len(tmp_json_data))):
            data_i = tmp_json_data[i]
            if data_i['id'] not in ppl_ins_dict.keys() or data_i['id'] not in emb_dict.keys():
                print(i)
                continue
            else:
                json_data.append(data_i)

        for i in tqdm(range(len(json_data))):
            data_i = json_data[i]
            if data_i['id'] not in ppl_ins_dict.keys():
                continue
            elif data_i['id'] not in emb_dict.keys():
                continue

            ppl_ins = ppl_ins_dict[data_i['id']][0].item()
            sent_emb = emb_dict[data_i['id']][0]

            loss_A=loss_dict[data_i['id']][0]
            loss_A1=loss_dict[data_i['id']][1]

            mean_atten_A=mean_atten_dict[data_i['id']][0]
            mean_atten_A1=mean_atten_dict[data_i['id']][1]
            max_atten_A=max_atten_dict[data_i['id']][0]
            max_atten_A1=max_atten_dict[data_i['id']][1]

            if args.model_name.startswith("baichuan"):
                prompt=baichuan2_template.replace("{question}", data_i['instruction'])
                original = data_i['output']+"</s>"
                generated = generated_output_dict[data_i['id']] + "</s>"
            else:
                prompt = qwen_template
                prompt = prompt.replace("{question}", data_i['instruction'])
                original = data_i['output'] + "<|im_end|>"
                generated = generated_output_dict[data_i['id']] + "<|im_end|>"

            mean_weighted_loss_A=weighted_loss(tokenizer,nlp,prompt+original,original,loss_A,mean_atten_A,args.max_length)
            mean_weighted_loss_A1=weighted_loss(tokenizer,nlp,prompt+generated,generated,loss_A1,mean_atten_A1,args.max_length)
            max_weighted_loss_A = weighted_loss(tokenizer, nlp, prompt + original, original, loss_A, max_atten_A,args.max_length)
            max_weighted_loss_A1 = weighted_loss(tokenizer, nlp, prompt + generated, generated, loss_A1, max_atten_A1,args.max_length)

            if np.isnan(mean_weighted_loss_A)  or np.isnan(max_weighted_loss_A) or np.isnan(ppl_ins) or np.isinf(ppl_ins) or np.isnan(mean_weighted_loss_A1) or np.isnan(max_weighted_loss_A1):
                continue
            else:
                mean_weighted_loss_A_list.append(mean_weighted_loss_A)
                mean_weighted_loss_A1_list.append(mean_weighted_loss_A1)
                max_weighted_loss_A_list.append(max_weighted_loss_A)
                max_weighted_loss_A1_list.append(max_weighted_loss_A1)
                emb_list.append(sent_emb)
                ppl_ins_list.append(ppl_ins)
                final_json_data.append(data_i)
        print(len(final_json_data))

        emb_vectors = torch.cat(emb_list, 0).to(torch.float16).numpy()
        mean_weighted_loss_A_array = np.array(mean_weighted_loss_A_list)
        mean_weighted_loss_A1_array = np.array(mean_weighted_loss_A1_list)
        max_weighted_loss_A_array = np.array(max_weighted_loss_A_list)
        max_weighted_loss_A1_array = np.array(max_weighted_loss_A1_list)
        ppl_ins_array = np.array(ppl_ins_list)

        torch.save({
            'emb_vectors': emb_vectors,
            'mean_weighted_loss_A_array':mean_weighted_loss_A_array,
            'mean_weighted_loss_A1_array':mean_weighted_loss_A1_array,
            'max_weighted_loss_A_array':max_weighted_loss_A_array,
            'max_weighted_loss_A1_array':max_weighted_loss_A1_array,
            'ppl_ins_array':ppl_ins_array,
            'final_json_data':final_json_data
        },args.analysis_save_path)

    if args.atten_method=='mean':
        weighted_loss_A_array=mean_weighted_loss_A_array
        weighted_loss_A1_array=mean_weighted_loss_A1_array
    elif args.atten_method=='max':
        weighted_loss_A_array=max_weighted_loss_A_array
        weighted_loss_A1_array=max_weighted_loss_A1_array

    lower_threshold_ins = np.percentile(ppl_ins_array,args.low_th)
    upper_threshold_ins = np.percentile(ppl_ins_array,args.up_th)

    lower_threshold_A = np.percentile(weighted_loss_A_array, args.low_th)
    upper_threshold_A = np.percentile(weighted_loss_A_array, args.up_th)

    lower_threshold_A1 = np.percentile(weighted_loss_A1_array, args.low_th)
    upper_threshold_A1 = np.percentile(weighted_loss_A1_array, args.up_th)

    # Get the indices of the samples within the middle level confidence range
    indices=np.array(range(len(final_json_data)))
    middle_indices = indices[
        (ppl_ins_array >= lower_threshold_ins) & (ppl_ins_array <= upper_threshold_ins)&
        (weighted_loss_A_array >= lower_threshold_A) & (weighted_loss_A_array <= upper_threshold_A) &
        (weighted_loss_A1_array >= lower_threshold_A1) & (weighted_loss_A1_array <= upper_threshold_A1)]

    def get_json_sample(data,samples):
        json_samples = []
        ids_list = samples.tolist()
        for id_i in ids_list:
            ori_sample = data[id_i]
            json_samples.append(ori_sample)

        return json_samples

    print(len(middle_indices))
    if len(middle_indices) <args.sample_num:
        new_data=get_json_sample(final_json_data,middle_indices)
    else:
        middle_confidence_embeds = emb_vectors[middle_indices]
        k_center = KCenterGreedy(middle_confidence_embeds)
        already_selected = None
        result = k_center.select_batch(already_selected, args.sample_num)
        middle_confidence_samples=middle_indices[result]
        new_data = get_json_sample(final_json_data,middle_confidence_samples)

    print('New data len \n', len(new_data))
    with open(args.json_save_path, "w") as fw:
        json.dump(new_data, fw, indent=4,ensure_ascii=False)


if __name__ == '__main__':
    main()