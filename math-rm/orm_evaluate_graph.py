from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
from accelerate import Accelerator
import numpy as np
import torch
from tqdm import tqdm
import argparse
import json
import time
import os
import sys
import re
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='pwork7/llama31_it_prm_2e6_bz32_1epoch_conversation')  # model path
    parser.add_argument("--tokenizer_path", type =str, default = 'Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument("--dataset", type=str, default='RLHFlow/Mistral-MATH500-Test')  # data path
    parser.add_argument("--output_dir", type=str, default="math_best_of_n")  # output dir
    parser.add_argument("--num_n", type=int, default=1024)  # number of N for each question
    parser.add_argument("--model_type",type=str,choices=["Mistral","Deepseek"],default='Mistral')
    parser.add_argument("--peft",type = bool, default = False)
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()
    return args 
def batch_data(data_list, batch_size=1):
    n = batch_size
    batch_data = []
    for i in range(n-1):
        start = i * (len(data_list) // batch_size)
        end = (i+1)* (len(data_list) // batch_size)
        batch_data.append(data_list[start:end])

    last_start = (n-1) * (len(data_list) // batch_size)
    batch_data.append(data_list[last_start:len(data_list)])
    return batch_data

def select_sample(args,sample,model,tokenizer,candidate_tokens,local_rank):
    prompt = sample['prompt']
    scores_list = []
    answers = sample['answers'][:args.num_n]
    
    step_scores = []
    for ans in answers:
        single_step_score = []
        conversation = []
        forward_conv = []
        if args.model_type == "Mistral":
            processed_ans = ans.replace(" ки","")
            conversation.append({"content":prompt + " " + processed_ans,"role":"user"})
        else:
            conversation.append({"content":prompt + " " + ans,"role":"user"})
        conversation.append({"content":"+","role":"assistant"})
        senten = tokenizer.apply_chat_template(conversation,return_tensors="pt", tokenize = False)

        # print(senten)
        input_ids = tokenizer.apply_chat_template(conversation,return_tensors="pt").to(local_rank)
        with torch.no_grad():
            logits = model(input_ids).logits[:,-3,candidate_tokens] #simple version for llama3.1-instruct, the +/- is predicted by the '-3' position
            scores = logits.softmax(dim=-1)[:,0] # 0 means the prob of + (1 mean -)

        scores_list.append(scores[0].detach().to('cpu', dtype=torch.float32))
        
    idx = scores_list.index(max(scores_list))
    sample['step_scores'] = [x.item() for x in scores_list]  # Add the step_score attribute to each sample
    return sample['label'][idx] == 1,sample

def select_sample2(args,sample,model,tokenizer,candidate_tokens,local_rank):
    prompt = sample['prompt']
    question = prompt
    scores_list = []
    answers = sample['answers'][:args.num_n]
    
    step_scores = []
    for ans in answers:
        single_step_score = []
        conversation = []
        forward_conv = []
        message = [
                {"role":"user", "content":question},
                {"role":"assistant", "content":ans}
            ]
        
        senten =tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
                # message, add_generation_prompt=False)
        input_ids = tokenizer(senten, truncation=False, return_tensors="pt").to(local_rank)

        # print(senten)
        # input_ids = tokenizer.apply_chat_template(message,return_tensors="pt").to(local_rank)
        # print(input_ids)
        #

        
        with torch.no_grad():
            outputs = model(
            # input_ids= input_ids,)
            input_ids= input_ids['input_ids'],
            attention_mask= input_ids['attention_mask'],
            )
            scores = torch.sigmoid(outputs.logits) # 0 means the prob of + (1 mean -)
            # print(scores)
            # print(scores.shape)
        scores_list.append(scores[0].detach().to('cpu', dtype=torch.float32))
        
    idx = scores_list.index(max(scores_list))
    picked_sample = sample['answers'][idx]
    sample['step_scores'] = [x.item() for x in scores_list]  # Add the step_score attribute to each sample
    # print(sample['label'][idx])
    return sample['label'][idx] == 1,picked_sample


def worker(args, model, tokenizer, data, local_rank):

    temp_instances = []
    plus_tag_id = tokenizer.encode('+')[-1]
    minus_tag_id = tokenizer.encode('-')[-1]
    candidate_tokens = [plus_tag_id,minus_tag_id]
    for i,sample in enumerate(tqdm(data)):
        sign,new_sample = select_sample2(args,sample,model,tokenizer,candidate_tokens,local_rank)
        data[i] = new_sample
        temp_instances.append(sign)
        if i % 50 == 0:
            print(f"Accuracy so far: {sum(temp_instances)/len(temp_instances)}")
    # Save results
    return temp_instances,data
       
if __name__ == "__main__":
    args = parse_args()

    accelerator = Accelerator()
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    #print(world_size)
    ds = load_dataset(args.dataset,split="train")
    if args.num_samples is not None:
        ds = ds.select(range(args.num_samples))
    local_rank = Accelerator().local_process_index
    print("---------------")
    print("begin to load reward model.")
    print("---------------")
    downloaded = False
    cnt = 0
    while not downloaded:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            if args.peft:
                base_model = AutoModelForSequenceClassification.from_pretrained(args.tokenizer_path, num_labels = 1, torch_dtype=torch.float16, device_map={'': torch.cuda.current_device()})
                # base_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-mini-instruct", torch_dtype=torch.bfloat16, device_map = torch.cuda.current_device()).to(local_rank).eval()
                base_model.config.pad_token_id = tokenizer.pad_token_id
                base_model.config.use_cache = False
                base_model.resize_token_embeddings(len(tokenizer))
                model = PeftModel.from_pretrained(base_model, args.model_path)
            else:
                
                # model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(local_rank).eval()
                model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels = 1, torch_dtype=torch.float16, device_map={'': torch.cuda.current_device()})
                model.config.use_cache = False
                model.config.pad_token_id = tokenizer.pad_token_id
                model.resize_token_embeddings(len(tokenizer))
                
            
            downloaded = True
        except Exception as error:
            print("An error occurred:", error)
            print("Failed to load the reward model. Retrying....")
            time.sleep(2)
            cnt+=1
            if cnt > 5:
                print("Failed to load the reward model. Please check the model path and try again.")
                sys.exit(1)

    tokenizer.padding_side = "right"
    # tokenizer.pad_token = tokenizer.eos_token
    # print('pad token:', tokenizer.pad_token)
    # print('pad token id:', tokenizer.pad_token_id)
    # model.config.pad_token_id = model.config.eos_token_id
    # print('pad token id:', model.config.pad_token_id)


    data = []
    data_size = len(ds["prompt"])

    share = int(data_size / world_size) + 1
    ds = ds.select(np.arange(local_rank * share, min((local_rank + 1) * share, len(ds))))
    print(ds)
    for sample in ds:
        data.append(sample)

    selected_data_label, new_data = worker(args,model,tokenizer,data,local_rank)
    
    # Send the data to other GPUs
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    all_process_list = [{}] * world_size

    data_to_send = {
        "data": [[selected_data_label[i]] for i in range(len(selected_data_label))],
        "new_data": [[new_data[i]] for i in range(len(new_data))]
    }
    # print(all_process_list)
    # print('----')
    # print(data_to_send['new_data'][0][0].keys())
    # print(data_to_send["data"][1][0])
    
    label = 'label'
    # import torch.distributed as dist
    # dist.init_process_group(backend="nccl")
    # dist.all_gather_object(all_process_list, data_to_send)
    # gathered_data = []
    # gathered_save_data = []

    # for i in range(world_size):
    #     tmp_data = [tmp[0] for tmp in data_to_send[i]["data"]]
    #     gathered_data.extend(tmp_data)
        
    #     tmp_save_data = [tmp[0] for tmp in data_to_send[i]["new_data"]]
    #     gathered_save_data.extend(tmp_save_data)
    data_to_send['data'] = [i[0] for i in data_to_send['data']]
    
    if local_rank == 0:
        print(f"acc: {sum(data_to_send['data'])/len(data_to_send['data'])}")
        acc = {"accuracy":sum(data_to_send['data'])/len(data_to_send['data'])}
    
        with open(f"{args.output_dir}_{args.num_n}_save_data.json",'w') as f: # We also save a copy of the step score.
            for entry in data_to_send["new_data"]:
                f.write(json.dumps(entry) + "\n")
        

