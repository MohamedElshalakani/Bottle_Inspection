import requests

url = "https://raw.githubusercontent.com/ultralytics/assets/main/yolov8n.onnx"
response = requests.get(url)

with open("yolov8n.onnx", "wb") as f:
    f.write(response.content)

print("Download complete! File size:", len(response.content), "bytes")
