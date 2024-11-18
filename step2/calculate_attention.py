import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import argparse
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_length=1024

model_name_or_path = "../../llm/hub/Qwen/Qwen1.5-7B-Chat"
generated_path="../data/qwen1_5_7b/qwen1_5_7b_high_quality_generated.jsonl"
output_path ='../data/qwen1_5_7b/meddata_qwen1_5_7b_atten.pt'
model_name="qwen"


baichuan2_template="""<s><reserved_106>{question}<reserved_107>"""

qwen_template="""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant\n"""


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path",type=str, default=model_name_or_path)
    parser.add_argument("--output_path", type=str,default=output_path)
    parser.add_argument("--generated_path",type=str,default=generated_path)
    parser.add_argument('--model_name',type=str,default=model_name)
    parser.add_argument('--max_length', type=int, default=max_length)

    return parser.parse_args()

def answer_perplexity_and_attention(tokenizer, model, text, target_span, max_length):

    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    # find the target span in the input ids
    start_index = text.rfind(target_span)
    start_token = len(tokenizer.encode(text[:start_index]))

    labels = input_ids.clone()
    labels[0, :start_token] = -100

    # Only calculate the losses of the answer part
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
    loss = outputs.loss
    perplexity = torch.exp(loss)
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    loss_fct = nn.CrossEntropyLoss(reduction='none')
    per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # restore the shape of the losses
    losses = per_token_loss.view(-1).tolist()

    # calculate aggregated attention
    with torch.no_grad():
        atten = model(input_ids, output_attentions=True).attentions[-1][0]
    atten=atten.detach().cpu()
    max_atten, _ = torch.max(atten, dim=1)
    max_atten = torch.mean(max_atten, dim=0)

    mean_atten = torch.sum(atten, dim=1)
    mean_atten = torch.mean(mean_atten, dim=0)
    for i in range(mean_atten.shape[0]):
        mean_atten[i] /= (mean_atten.shape[0] - i)
    mean_atten=mean_atten.tolist()
    max_atten=max_atten.tolist()
    torch.cuda.empty_cache()

    return max_atten,mean_atten ,perplexity.to('cpu'), losses

def main():
    args = parse_arguments()

    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True,
                                                 torch_dtype=torch.float16, device_map='auto')
    model.to(device)
    model.eval()

    continue_id = set()
    if os.path.exists(args.output_path):
        pre_result = torch.load(args.output_path, map_location=torch.device('cpu'))
        for data in pre_result:
            continue_id.add(data['id'])
    else:
        pre_result = []

    datas = []
    with open(args.generated_path, 'r', encoding='utf-8') as f:
        for line in f:
            data=json.loads(line)
            if data['id'] in continue_id:
                continue
            else:
                datas.append(data)

    new_datas=[]
    cnt=1
    for i in tqdm(range(len(datas))):
        data = datas[i]

        question = data['instruction']
        answer = data['output']
        generated_output = data['generated_output']
        if model_name.startswith("baichuan"):
            prompt = baichuan2_template.replace("{question}", question)
            original = answer + "</s>"
            generated = generated_output + "</s>"
        else:
            prompt = qwen_template
            prompt = prompt.replace("{question}", question)
            original = answer + "<|im_end|>"
            generated = generated_output + "<|im_end|>"

        max_atten_A, mean_atten_A, ppl_A, loss_A = answer_perplexity_and_attention(tokenizer, model, prompt + original,
                                                                                   original, args.max_length)
        max_atten_A1, mean_atten_A1, ppl_A1, loss_A1 = answer_perplexity_and_attention(tokenizer, model,
                                                                                       prompt + generated, generated,
                                                                                       args.max_length)

        data['ppl']=[ppl_A,ppl_A1]
        data['loss']=[loss_A,loss_A1]
        data['max_token_atten']=[max_atten_A,max_atten_A1]
        data['mean_token_atten']=[mean_atten_A,mean_atten_A1]

        # save itermediate results
        new_datas.append(data)
        cnt+=1
        if cnt%1000==0:
            torch.save(new_datas+pre_result,args.output_path)

    torch.save(new_datas+pre_result,args.output_path)

if __name__=="__main__":
    main()