import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import argparse
import json
import logging
import pathlib
import random
import sys
from copy import deepcopy
import torch
from tqdm import tqdm
from transformers import AutoTokenizer,LlamaTokenizer
from xopen import xopen
from vllm import LLM, SamplingParams

from utils import *
logger = logging.getLogger(__name__)
random.seed(42)
# deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# meta-llama/Llama-3.1-8B

def tokenizer_instance(ins,chat_tokenizer,distill=False):
    if not ins["documents"]:
        prompt = f"Question: {ins['question']}\nAnswer:"
        ins["prompt"] = prompt
        return ins
    system_prompt = "You are an intelligent AI assistant. Please answer questions based on the user's instructions. Below are some reference documents that may help you in answering the user's question.\n\n"
    for d_idx in range(0, len(ins['documents'])):
        doc = ins["documents"][d_idx]
        system_prompt += f"- Title: {doc['title']}\n{doc['text'].strip()}\n"
    system_prompt = system_prompt.strip()

    user_prompt = f"Please write a high-quantify answer for the given question using only the provided search documents (some of which might be irrelevant).\nQuestion: {ins['question']}".strip()
    if distill:
        prompt = chat_tokenizer.apply_chat_template(
            conversation=[
                # {"role": "system", "content": system_prompt},
                {"role": "user", "content": system_prompt + user_prompt}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        prompt = f"{system_prompt}### Instruction:\n{user_prompt}\nAnswer:\n### Response:\n"
    

    ins["prompt"] = prompt
    # ins["prompt"] = prompt.replace(
    #     "<|eot_id|><|start_header_id|>user<|end_header_id|>",
    #     "\n<|eot_id|><|start_header_id|>user<|end_header_id|>"
    # )
    return ins


def main(
    input_path,
    model_name,
    temperature,
    top_p,
    closedbook,
    prompt_mention_random_ordering,
    num_gpus,
    max_new_tokens,
    max_prompt_length,
    output_path,
    batch_size,
    sample_num,
    gold_doc
): 
    base_model = model_name.split("/")[-1]

    chat_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1") if "distill" in base_model.lower() else AutoTokenizer.from_pretrained(f"{model_name}-Instruct")
    Distill = True if "distill" in base_model.lower() else False
    logger.info(f"Wether distilled models:{Distill}") 


    if closedbook:
        output_path = os.path.join(output_path,f"{base_model}_closebook_{max_new_tokens}.jsonl.gz")
    else:
        output_path = os.path.join(output_path,f"{base_model}_{gold_doc}.jsonl.gz")
    if os.path.exists(output_path):
        print("-"*45)
        print("Response has been generated!")
        print("-"*45)
        evaluate_qa_data(input_path=output_path)
        return


    # Create directory for output path if it doesn't exist.
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    examples = []
    prompts = []


    # Fetch all of the prompts
    with xopen(input_path) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            # Get the prediction for the input example

            if closedbook:
                input_example["documents"] = []
            else:
                documents = []
                gold_document = input_example["gold_document"]
                distractors = input_example["distractors"]
                distractors.insert(gold_doc, gold_document)
                documents_dict = distractors
                for ctx in deepcopy(documents_dict):
                    documents.append(Document.from_dict(ctx))
                input_example["documents"] = documents_dict
                if not documents:
                    raise ValueError(f"Did not find any documents for example: {input_example}")
            # breakpoint()
            prompt = tokenizer_instance(input_example,chat_tokenizer,distill=Distill)["prompt"]
            
            prompts.append(prompt)
            examples.append(deepcopy(input_example))
            # all_model_documents.append(documents)
            if len(prompts) == sample_num:
                break
            
    model = LLM(
            model=model_name,
            tensor_parallel_size=num_gpus,
            load_format="safetensors",
            max_num_batched_tokens=max_prompt_length,
        )
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)
    raw_responses = model.generate(prompts, sampling_params)
    responses = [output.outputs[0].text.strip() for output in raw_responses]

            

    logger.info(f"Loaded {len(prompts)} prompts to process")

    # Get responses for all of the prompts
    if not torch.cuda.is_available():
        raise ValueError("Unable to find CUDA device with torch. Please use a CUDA device to run this script.")
    


    
    
    
    with xopen(output_path, "w") as f:
        for example, prompt, response in zip(examples, prompts, responses):
            output_example = deepcopy(example)
            # Add some extra metadata to the output example
            output_example["question"] = example["question"]
            # output_example["model_documents"] = [dataclasses.asdict(document) for document in model_documents]
            output_example["model_answer"] = response

            f.write(json.dumps(output_example) + "\n")
    evaluate_qa_data(input_path=output_path)

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", help="Path to data with questions and documents to use.", required=True)
    parser.add_argument(
        "--model",
        help="Model to use in generating responses",
        required=True,
    )
    parser.add_argument("--temperature", help="Temperature to use in generation", type=float, default=0.6)
    parser.add_argument("--top-p", help="Top-p to use in generation", type=float, default=1.0)
    parser.add_argument(
        "--closedbook", action="store_true", help="Run the model in closed-book mode (i.e., don't use documents)."
    )
    parser.add_argument(
        "--prompt-mention-random-ordering",
        action="store_true",
        help="Mention that search results are ordered randomly in the prompt",
    )
    parser.add_argument("--num-gpus", help="Number of GPUs to use", type=int, default=1)
    parser.add_argument("--hf-cache-path", help="Path to huggingface cache to use.")
    parser.add_argument("--output-path", help="Path to write output file of generated responses", required=True)
    parser.add_argument(
        "--max-new-tokens",
        help="Maximum number of new tokens to generate",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--max-prompt-length",
        help="Maximum number of tokens in the prompt. Longer prompts will be skipped.",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--batch_size",
        help="batch size of inference",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--sample_num",
        help="sample size",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--top_p",
        help="top_p",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--gold_doc",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))
    main(
        args.input_path,
        args.model,
        args.temperature,
        args.top_p,
        args.closedbook,
        args.prompt_mention_random_ordering,
        args.num_gpus,
        args.max_new_tokens,
        args.max_prompt_length,
        args.output_path,
        args.batch_size,
        args.sample_num,
        args.gold_doc
    )
    logger.info("finished running %s", sys.argv[0])
