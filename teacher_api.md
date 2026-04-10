  AI 模型 API 整合手冊

  本服務提供完全兼容 OpenAI 規範的 API 介面，支援純文字對話 (Text)、多模態視覺分析 (Image/Vision) 以及文字特徵向量生成 (Embedding)。

  1. 服務基礎資訊

  * API 金鑰 (Bearer Token): YOUR_API_KEY_HERE
   * 授權方式: HTTP Header 帶入 Authorization: Bearer <API_KEY>
   * 資料格式: Content-Type: application/json

  ┌──────────────────────────┬──────────────────────────┬───────────────────────────────┐
  │ 服務類型                 │ 模型名稱 (Model)         │ Base URL                      │
  ├──────────────────────────┼──────────────────────────┼───────────────────────────────┤
  │ 文字與視覺 (Chat/Vision) │ gemma-4-31B-it-FP8-block │ http://<SERVER_IP>:8084/v1    │
  │ 向量生成 (Embedding)     │ bge-m3                   │ http://<SERVER_IP>:8085/v1    │
  └──────────────────────────┴──────────────────────────┴───────────────────────────────┘
  ---

  2. 快速起步：使用官方 OpenAI SDK (Python)

  因為本 API 完全兼容 OpenAI 協議，您不需要重新撰寫底層請求，只需將官方 SDK 的 base_url 與 api_key 指向本伺服器即可。

  安裝套件
   pip install openai

  初始化客戶端

   from openai import OpenAI

   # 建立 Chat/Vision 客戶端 (Port 8084)
   client_chat = OpenAI(
      base_url="http://<SERVER_IP>:8084/v1",
      api_key="YOUR_API_KEY_HERE"
   )

   # 建立 Embedding 客戶端 (Port 8085)
   client_embed = OpenAI(
      base_url="http://<SERVER_IP>:8085/v1",
      api_key="YOUR_API_KEY_HERE"
   )

  ---

  3. 功能範例說明

  📝 情境 A：純文字對話 (Text Completions)
   * 端點: POST /chat/completions (使用 Port 8084)
   * 說明: 標準的多輪對話生成，支援 system, user, assistant 角色。

  Python (OpenAI SDK)

   response = client_chat.chat.completions.create(
       model="gemma-4-31B-it-FP8-block",
       messages=[
           {"role": "system", "content": "你是一位專業的程式助手。"},
           {"role": "user", "content": "請解釋什麼是 RESTful API？"}
       ],
       max_tokens=500
   )
   print(response.choices[0].message.content)

  cURL (HTTP 請求)

   curl -X POST http://<SERVER_IP>:8084/v1/chat/completions \
     -H "Authorization: Bearer YOUR_API_KEY_HERE" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "gemma-4-31B-it-FP8-block",
       "messages": [
         {"role": "user", "content": "請解釋什麼是 RESTful API？"}
       ]
     }'

  ---

  🖼️ 情境 B：多模態視覺分析 (Image / Vision)
   * 端點: POST /chat/completions (使用 Port 8084)
   * 說明: 允許傳入圖片網址，模型將根據圖片內容回答問題。
   * 注意事項: 傳入的 url 必須是可直接下載的圖片真實連結 (例如結尾為 .png, .jpg)，不可為含有防爬蟲機制的網頁連結。

  Python (OpenAI SDK)

   response = client_chat.chat.completions.create(
       model="gemma-4-31B-it-FP8-block",
       messages=[
           {
               "role": "user",
               "content": [
                   {"type": "text", "text": "這張圖片是什麼？請簡短回答。"},
                   {
                       "type": "image_url",
                       "image_url": {
                           "url": "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_92x30dp.png"
                       }
                   }
               ]
           }
       ],
       max_tokens=300
   )
   print(response.choices[0].message.content)

  cURL (HTTP 請求)

   curl -X POST http://<SERVER_IP>:8084/v1/chat/completions \
     -H "Authorization: Bearer YOUR_API_KEY_HERE" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "gemma-4-31B-it-FP8-block",
       "messages": [{
         "role": "user",
         "content": [
           {"type": "text", "text": "這張圖片是什麼？請簡短回答。"},
           {"type": "image_url", "image_url": {"url": "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_92x30dp.png"}}
         ]
       }]
     }'

  ---

  🧮 情境 C：文字向量生成 (Embeddings)
   * 端點: POST /embeddings (使用 Port 8085)
   * 說明: 用於語意搜尋 (Semantic Search) 或檢索增強生成 (RAG)。將輸入文字轉換為 1024 維的浮點數向量。

  Python (OpenAI SDK)

   response = client_embed.embeddings.create(
       model="bge-m3",
       input="今天天氣真好，適合出門散步。"
   )
   # 取得浮點數陣列
   vector = response.data[0].embedding
   print(f"向量維度大小: {len(vector)}")

  cURL (HTTP 請求)

   curl -X POST http://<SERVER_IP>:8085/v1/embeddings \
     -H "Authorization: Bearer YOUR_API_KEY_HERE" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "bge-m3",
       "input": "今天天氣真好，適合出門散步。"
     }'

  ---

  4. 常見問題與排錯 (Troubleshooting)

   1. 圖片無法解析 (400 Bad Request / 500 Internal Error)
      * 請確認圖片 url 是否為直連圖檔 (Direct Link)。若是受權限保護的網址（如 Google 雲端硬碟分享連結、部分社群媒體圖片），伺服器將無法下載。建議在實際應用中，由後端將圖片轉為 base64 格式後再送出。
   2. JSON 解析錯誤 (Invalid control character at...)
      * 若使用 curl 測試，請確保 JSON 字串內沒有包含隱藏的換行字元。
   3. 連線超時 (Timeout)
      * 若傳送超長文本，請確認客戶端的 HTTP Request Timeout 設定時間足夠（建議設為 60 秒以上）。上下文限制最高支援 65,536 Tokens。