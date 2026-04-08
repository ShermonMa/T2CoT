DATA_DIR="/root/autodl-tmp/T2CoT/data"
OUTPUT_DIR="/root/autodl-tmp/T2CoT/output"
datasets=(
  "$DATA_DIR/synthcypher_test/question.txt $DATA_DIR/synthcypher_test/generate_schema.json synthcypher_test"
)

PYTHON_SCRIPT="/root/autodl-tmp/T2CoT/experiment/run_gpt_test_optimized.py"
COMMON_ARGS="--run --model gpt-3.5-turbo --temperature 0 --max_tokens 300"

for data in "${datasets[@]}"; do
  read -r question schema output <<< "$data"
  echo "==$question==$schema== $output ======"
  python "$PYTHON_SCRIPT" \
         --run \
         --question "$question" \
         --schema   "$schema" \
         --out      "${OUTPUT_DIR}/${output}/gpt_cot.jsonl" \
         --cot          \
         --start_from 3967\
         $COMMON_ARGS
  echo "Done -> $output"
done
for data in "${datasets[@]}"; do
  read -r question schema output <<< "$data"
  echo "==$question==$schema== $output ======"
  python "$PYTHON_SCRIPT" \
         --run \
         --question "$question" \
         --schema   "$schema" \
         --out      "${OUTPUT_DIR}/${output}/gpt_noCot.jsonl" \
         $COMMON_ARGS
  echo "Done -> $output"
done