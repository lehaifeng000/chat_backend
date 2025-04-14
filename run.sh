conda activate SVEMath
CUDA_VISIBLE_DEVICES="0" nohup uvicorn main:app --reload --port 8080 &