import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import math
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import dataclasses
import json
import logging
import pathlib
import random
import sys
from copy import deepcopy
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from xopen import xopen
from transformers.models.llama import modeling_llama
from transformers.models.mistral import modeling_mistral
from vllm import LLM, SamplingParams
from prompting import (
    Document,
    get_closedbook_qa_prompt,
    get_qa_prompt,
)
from utils import *
logger = logging.getLogger(__name__)
random.seed(42)
cache_dir="/root/hugging_cache"



def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def format_instruct_prompt(instruction):
    INSTRUCTION_KEY = "### Instruction:"
    RESPONSE_KEY = "### Response:"
    INTRO_BLURB = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    )
    PROMPT_FOR_GENERATION = "{intro}\n{instruction_key}\n{instruction}\n{response_key}\n".format( #<think>\n
        intro=INTRO_BLURB,
        instruction_key=INSTRUCTION_KEY,
        instruction=instruction,
        response_key=RESPONSE_KEY,
    )
    return PROMPT_FOR_GENERATION


def main(
    input_path,
    model_name,
    temperature,
    top_p,
    closedbook,
    prompt_mention_random_ordering,
    use_random_ordering,
    query_aware_contextualization,
    num_gpus,
    max_new_tokens,
    max_prompt_length,
    hf_cache_path,
    output_path,
    batch_size,
    sample_num,
    gold_doc
): 
    if closedbook:
        output_path = os.path.join(output_path,f"{model_name}_closebook_{max_new_tokens}.jsonl.gz")
    else:
        output_path = os.path.join(output_path,f"{model_name}_{input_path.split('/')[-1]}")
    if os.path.exists(output_path):
        print("-"*45)
        print("Response has been generated!")
        print("-"*45)
        evaluate_qa_data(input_path=output_path)
        return


    # Create directory for output path if it doesn't exist.
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    hf_cache_path = os.path.join(cache_dir,model_name)
    tokenizer = AutoTokenizer.from_pretrained(hf_cache_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token  # to avoid an error
    examples = []
    prompts = []
    all_model_documents = []
    did_format_warn = False
    # Fetch all of the prompts
    with xopen(input_path) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            # Get the prediction for the input example
            question = input_example["question"]
            if closedbook:
                documents = []
            else:
                documents = []
                gold_document = input_example["gold_document"]
                distractors = input_example["distractors"]
                distractors.insert(gold_doc, gold_document)
                documents_dict = distractors
                for ctx in deepcopy(documents_dict):
                    documents.append(Document.from_dict(ctx))
                if not documents:
                    raise ValueError(f"Did not find any documents for example: {input_example}")

            

            if closedbook:
                prompt = get_closedbook_qa_prompt(question)
            else:
                prompt = get_qa_prompt(
                    question,
                    documents,
                    mention_random_ordering=prompt_mention_random_ordering,
                    query_aware_contextualization=query_aware_contextualization,
                )

            if ("chat" in model_name.lower()) or ("instruct") in model_name.lower():
                if did_format_warn is False:
                    logger.warning(f"Model {model_name} appears to be an chat model, applying chat formatting")
                    did_format_warn = True
            prompt = format_instruct_prompt(prompt)
            prompt_length = len(tokenizer(prompt)["input_ids"])
            if max_prompt_length < prompt_length:
                logger.info(
                    f"Skipping prompt {prompt[:100]}... with length {prompt_length}, which "
                    f"is greater than maximum prompt length {max_prompt_length}"
                )
                continue
            if len(prompts) == sample_num:
                break
            prompts.append(prompt)
            examples.append(deepcopy(input_example))
            all_model_documents.append(documents)
    logger.info(f"Loaded {len(prompts)} prompts to process")

    # Get responses for all of the prompts
    if not torch.cuda.is_available():
        raise ValueError("Unable to find CUDA device with torch. Please use a CUDA device to run this script.")
    
    logger.info("Loading tokenizer")
    logger.info("Loading model")
    
    model = AutoModelForCausalLM.from_pretrained(hf_cache_path,torch_dtype=torch.float32,device_map="auto")
    responses=[]
    #inference at batch
    
    for batched_prompts in tqdm(chunks(prompts,batch_size)):
        input = tokenizer(batched_prompts, add_special_tokens=False, return_tensors='pt', truncation=True, max_length=model.config.max_position_embeddings, padding=True).to(model.device)
        # breakpoint()
        with torch.no_grad():
            outputs = model.generate(
                **input,
                do_sample=True,
                temperature = temperature,
                # top_p = top_p,
                # top_p = top_p,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                return_dict_in_generate=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        # decode
            # breakpoint()
            batched_prompts_length = input.input_ids.size()[1]
            decoded_outputs= tokenizer.batch_decode(outputs[:,batched_prompts_length:],skip_special_tokens=True)
            # breakpoint()
            responses.extend(decoded_outputs)
        torch.cuda.empty_cache()

    
    with xopen(output_path, "w") as f:
        for example, model_documents, prompt, response in zip(examples, all_model_documents, prompts, responses):
            output_example = deepcopy(example)
            # Add some extra metadata to the output example
            output_example["model_prompt"] = prompt
            output_example["model_documents"] = [dataclasses.asdict(document) for document in model_documents]
            output_example["model_answer"] = response
            output_example["model"] = model_name
            output_example["model_temperature"] = temperature
            output_example["model_top_p"] = top_p
            output_example["model_prompt_mention_random_ordering"] = prompt_mention_random_ordering
            output_example["model_use_random_ordering"] = use_random_ordering
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
    parser.add_argument(
        "--use-random-ordering",
        action="store_true",
        help="Randomize the ordering of the distractors, rather than sorting by relevance.",
    )
    parser.add_argument(
        "--query-aware-contextualization",
        action="store_true",
        help="Place the question both before and after the documents.",
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

    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))
    main(
        args.input_path,
        args.model,
        args.temperature,
        args.top_p,
        args.closedbook,
        args.prompt_mention_random_ordering,
        args.use_random_ordering,
        args.query_aware_contextualization,
        args.num_gpus,
        args.max_new_tokens,
        args.max_prompt_length,
        args.hf_cache_path,
        args.output_path,
        args.batch_size,
        args.sample_num,
        args.gold_doc
    )
    logger.info("finished running %s", sys.argv[0])
