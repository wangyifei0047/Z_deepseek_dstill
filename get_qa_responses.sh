cd Z_deepseek_dstill
for GOLD_INDEX in 5 6 7 8; do
  INPUT_PATH="./qa_data/20_total_documents/nq-open-20_total_documents_gold_at_${GOLD_INDEX}.jsonl.gz"
  echo "Processing with gold-index=${GOLD_INDEX}..."

  # 运行 Python 脚本
  CUDA_VISIBLE_DEVICES=5 /root/miniconda3/envs/position_bias/bin/python ./src/get_qa_reponses.py \
    --input-path $INPUT_PATH \
    --model DeepSeek-R1-Distill-Llama-8B \
    --output-path results/rag\
    --batch_size 5 \
    --sample_num 500 \
    --temperature 0.6 \
    --max-prompt-length 32000 \
    --max-new-tokens 100 \
    # --closedbook 
    
    # --prompt-mention-random-ordering \
  echo "Finished processing gold-index=${GOLD_INDEX}"
done