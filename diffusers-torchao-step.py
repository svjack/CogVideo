https://github.com/sayakpaul/diffusers-torchao
->
https://github.com/sayakpaul/diffusers-torchao/blob/main/inference/benchmark_video.py

https://github.com/THUDM/CogVideo
->
https://github.com/THUDM/CogVideo/blob/main/inference/cli_demo_quantization.py

sudo apt-get update
sudo apt-get install git-lfs cbm ffmpeg

conda create --name py310 python=3.10
conda activate py310
pip install ipykernel
python -m ipykernel install --user --name py310 --display-name "py310"

git clone https://huggingface.co/spaces/svjack/CogVideoX-5B-Space
cd CogVideoX-5B-Space && pip install -r requirements.txt

'''
https://github.com/THUDM/CogVideo/issues/245
torch>=2.5.0
torchao==0.5

don't use float16, should use bfloat16

python t2v_app_qm.py --quantization_scheme int8wo --dtype bfloat16
python t2v_app_qm.py --quantization_scheme fp8wo --dtype bfloat16
python t2v_app_qm.py --quantization_scheme bf16 --dtype bfloat16
python t2v_app_qm.py --quantization_scheme fp16 --dtype float16

python i2v_app_qm.py --quantization_scheme int8wo --dtype bfloat16
python i2v_app_qm.py --quantization_scheme fp8wo --dtype bfloat16
python i2v_app_qm.py --quantization_scheme bf16 --dtype bfloat16
python i2v_app_qm.py --quantization_scheme fp16 --dtype float16

python v2v_app_qm.py --quantization_scheme int8wo --dtype bfloat16
python v2v_app_qm.py --quantization_scheme fp8wo --dtype bfloat16
python v2v_app_qm.py --quantization_scheme bf16 --dtype bfloat16
python v2v_app_qm.py --quantization_scheme fp16 --dtype float16
'''
