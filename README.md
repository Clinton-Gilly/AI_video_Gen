<div align="center">

# AI_video_Gen 🎬

### AI-Powered Short Video Generator

Provide a video **topic** or **keyword**, and AI_video_Gen will automatically generate the script, match footage, create subtitles and background music, and produce a high-quality short video — all in one click.

[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)](https://github.com/Clinton-Gilly/AI_video_Gen/releases/latest)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

---

## 📖 Overview

**AI_video_Gen** is a fully automated AI short video generation tool. It takes a topic or keyword as input and handles everything else:

- ✍️ **Script Generation** — Uses LLMs (GPT, Claude, Gemini, etc.) to write a compelling video script
- 🎥 **Footage Matching** — Automatically searches and selects relevant video clips from Pexels, Pixabay, and more
- 🗣️ **Text-to-Speech** — Generates professional voiceovers using Azure TTS, OpenAI TTS, and other providers
- 📝 **Subtitles** — Auto-generates and burns accurate subtitles into the video
- 🎵 **Background Music** — Adds fitting background music to enhance the final video
- 📱 **Multiple Formats** — Supports vertical (9:16), horizontal (16:9), and square (1:1) output

---

## 🎬 Example Videos

> These videos were generated entirely by AI_video_Gen — no manual editing.

### 🔐 How AI Helps in Security When Vibe Coding

https://github.com/Clinton-Gilly/AI_video_Gen/raw/main/docs/examples/ai-security-vibe-coding.mp4

> *Topic: "How AI helps in security when vibe coding" · Vertical 9:16 · Voice: en-AU-NatashaNeural · Source: Pexels*

---

### 💻 How AI Helps Software Developers Daily

https://github.com/Clinton-Gilly/AI_video_Gen/raw/main/docs/examples/ai-helps-developers.mp4

> *Topic: "How AI helps software developers daily" · Vertical 9:16 · Voice: en-US Neural · Source: Pexels*

---

## 🖥️ Screenshots

<h4 align="center">Web UI</h4>

![Web UI](docs/webui-en.jpg)

<h4 align="center">API Interface</h4>

![API](docs/api.jpg)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [FFmpeg](https://ffmpeg.org/download.html) installed and on your PATH
- API keys for at least one LLM provider (OpenAI, Claude, Gemini, etc.)

### Installation

```bash
# Clone the repository
git clone https://github.com/Clinton-Gilly/AI_video_Gen.git
cd AI_video_Gen

# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Copy the example config and fill in your API keys
cp config.example.toml config.toml
```

Open `config.toml` and set your keys:

```toml
[app]
# LLM provider: "openai" | "moonshot" | "azure" | "gemini" | ...
llm_provider = "openai"

[openai]
api_key = "sk-..."
model_name = "gpt-4o"

[pexels]
api_key = "your-pexels-api-key"
```

### Launch the Web UI

```bash
# Windows
webui.bat

# macOS / Linux
bash webui.sh

# Or run directly
python main.py
```

Open your browser at **http://localhost:8501**

---

## ⚙️ Features

| Feature | Description |
|---|---|
| 🤖 LLM Support | OpenAI, Claude, Gemini, DeepSeek, Moonshot, Azure, Ollama & more |
| 🎙️ TTS Engines | Azure TTS, OpenAI TTS, MiniMax, EdgeTTS, Fish Audio & more |
| 🎬 Video Sources | Pexels, Pixabay, local files |
| 📐 Aspect Ratios | 9:16 (vertical), 16:9 (horizontal), 1:1 (square) |
| 🌐 Languages | Multi-language script and subtitle support |
| 🖥️ Interfaces | Web UI (Streamlit) + REST API (FastAPI) |
| 🐳 Docker | Full Docker & Docker Compose support |

---

## 🐳 Docker

```bash
# Start with Docker Compose
docker-compose up -d

# GPU version
docker-compose -f docker-compose.gpu.yml up -d
```

---

## 🔌 API Usage

The project exposes a REST API via FastAPI. Once the server is running, visit:

```
http://localhost:8080/docs
```

for the interactive Swagger documentation.

---

## 📁 Project Structure

```
AI_video_Gen/
├── app/                # Core application logic
│   ├── models/         # Data models
│   ├── services/       # Business logic & AI integrations
│   └── utils/          # Shared utilities
├── webui/              # Streamlit web interface
├── docs/               # Documentation & screenshots
│   └── examples/       # Example generated videos
├── resource/           # Static resources (fonts, music, etc.)
├── storage/            # Generated output files (gitignored)
├── main.py             # Entry point
├── cli.py              # Command-line interface
└── config.example.toml # Configuration template
```

---

## 🛠️ CLI Usage

```bash
# Generate a video from the command line
python cli.py --subject "The Future of AI" --language en
```

Run `python cli.py --help` for all available options.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "feat: add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## ⭐ Star History

If you find this project useful, please consider giving it a star ⭐ — it helps a lot!
