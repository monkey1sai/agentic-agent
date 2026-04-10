  BGE-M3 Embedding API 使用範例

  服務位址：

  http://<SERVER_IP>:8085

  API Key：

  YOUR_API_KEY_HERE

  查模型：

  curl -sS \
    -H "Authorization: Bearer YOUR_API_KEY_HERE" \
    http://<SERVER_IP>:8085/v1/models

  單筆文字轉 embedding：

  curl -sS \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer YOUR_API_KEY_HERE" \
    http://<SERVER_IP>:8085/v1/embeddings \
    -d '{
      "model": "bge-m3",
      "input": "這是一段要轉成向量的文字"
    }'

  多筆文字轉 embedding：

  curl -sS \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer YOUR_API_KEY_HERE" \
    http://<SERVER_IP>:8085/v1/embeddings \
    -d '{
      "model": "bge-m3",
      "input": [
        "第一段文字",
        "第二段文字",
        "第三段文字"
      ]
    }'

  健康檢查：

  curl -sS http://<SERVER_IP>:8085/health

  Python 範例：

  import requests

  url = "http://<SERVER_IP>:8085/v1/embeddings"
  headers = {
      "Authorization": "Bearer YOUR_API_KEY_HERE",
      "Content-Type": "application/json",
  }
  payload = {
      "model": "bge-m3",
      "input": ["第一段文字", "第二段文字"]
  }

  resp = requests.post(url, headers=headers, json=payload, timeout=60)
  resp.raise_for_status()
  data = resp.json()

  print("model:", data["model"])
  print("items:", len(data["data"]))
  print("dims:", len(data["data"][0]["embedding"]))