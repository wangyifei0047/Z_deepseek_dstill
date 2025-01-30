
INPUT_PATH="./qa_data/20_total_documents.jsonl.gz"
# {0..19} 
for GOLD_INDEX in 0 ; do
  echo "Processing with gold-index=${GOLD_INDEX}..."

  # 运行 Python 脚本
  python ./src/get_qa_responses_vllm.py \
    --input-path $INPUT_PATH \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B\
    --output-path vllm_res \
    --batch_size 1 \
    --sample_num 100 \
    --temperature 0.6 \
    --max-prompt-length 32000 \
    --max-new-tokens 4096 \
    --gold_doc $GOLD_INDEX \
    --num_gpus 2 \
    --closedbook 
    
    # --prompt-mention-random-ordering \
  echo "Finished processing gold-index=${GOLD_INDEX}"
done