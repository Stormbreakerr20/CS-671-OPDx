# Install dependencies from requirements.txt
pip install -r UI/requirements.txt

# Install poppler-utils, tesseract-ocr and Ollama
sudo apt-get install poppler-utils tesseract-ocr

curl -fsSL https://ollama.com/install.sh | sh

ollama pull llava:7b-v1.5-q4_0
sleep 300

ollama pull medllama2:7b
sleep 300