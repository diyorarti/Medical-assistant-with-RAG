# RAG-API


Docker file runinning
```bash
docker run --rm -it `
  --env-file .env `
  -p 8000:8000 `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/hf-cache:/root/.cache/huggingface" `
  --name medrag medrag-api:latest
```