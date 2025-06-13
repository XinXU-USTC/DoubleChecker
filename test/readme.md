
## Benchmark Evaluation

### Data Preparation

All benchmark datasets for evaluation should be placed in the `./data` directory.

To add a new test dataset, follow the format of existing benchmarks in the `./data` directory.

### Prompt Configuration

For mathematical problems, we use the Qwen-instruct template:

```python
system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

few_shot_prompt = ""

question_format = """{question}"""
```

When adding a new mathematics benchmark, you can directly copy the above content to the corresponding `./prompts/qwen-instruct/xxx.py` file.


### Running Evaluation
You can run evaluation on "math" dataset using "qwen-instruct" prompt with greedy decoding, 32K max tokens, and 1 round of critique using four GPUs using the following command:

```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' \
python eval.py \
--model_name_or_path "/path/to/model/weights" \  # Path to model weights
--data_name "math" \  # Benchmark name (corresponding to first-level directory in ./data)
--prompt_type "qwen-instruct" \  # Default chat template
--temperature 0.0 \  # Sampling temperature
--start_idx 0 \  # Starting index for evaluation data
--end_idx -1 \  # Ending index for evaluation data
--n_sampling 1 \  # Number of samples per question
--k 1 \  # k value for unbiased pass@k calculation
--split "test" \  # Benchmark subset partition
--max_tokens 32768 \  # Maximum output length
--seed 0 \  # Random seed
--top_p 1 \  # Top-p sampling parameter
--surround_with_messages \  # Enable this flag if using chat template
--num_rounds 1 # number of rounds to self-critique
```

## Acknowledgments

Our evaluation code is modified from [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math/tree/main/evaluation). We thank their team for their valuable contributions to the community.