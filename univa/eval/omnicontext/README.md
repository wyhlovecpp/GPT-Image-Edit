The evaluation of the OmniContext benchmark includes the following steps:

## Step1 Environment Setup

```bash
pip install -U datasets megfile
```

## Step2 Generate Images


```
results/
├── {method_name}/
│   └── fullset/
│       └── {task_type}/
│           ├── key1.png
│           ├── key2.png
│           └── ...
```

To test OmniContext single set, you can run the following script to generate images:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun inference.py \
--nproc_per_node 8 \
--config omnicontext.yaml \
--test_data "OmniGen2/OmniContext"
```

##  Step3 Evaluation

1. We use GPT-4.1 to evaluate the quality of the generated images. Please make sure to set up your API key before running the script.

```bash
cd OmniGen2
python -m test_omnicontext_score \
--test_data "OmniGen2/OmniContext" \
--result_dir "/opt/dlami/nvme/wyh/code/fluxkontext_qwenvl/results_univorld_batch8/omnicontext" \
--model_name "univa_flux_omnicontext" \
--openai_key ${openai_key} \
--max_workers 100
```

2. Next, calculate the final score:

```bash
python -m calculate_statistics \
--save_path "/opt/dlami/nvme/wyh/code/fluxkontext_qwenvl/results_univorld_batch8/omnicontext" \
--model_name "univa_flux_omnicontext_fixed1024" \
--backbone gpt4dot1
```


## Acknowledgements

The code structure of this benchmark is inspired by [Step1X-Edit](https://github.com/stepfun-ai/Step1X-Edit).

Special thanks to the original project for their valuable contribution.