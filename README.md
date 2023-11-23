# YouDub: 优质视频中文化工具

## 简介
`YouDub` 是一个创新的开源工具，专注于将 YouTube 等平台的优质视频翻译和配音为中文版本。此工具融合了先进的 AI 技术，包括语音识别、大型语言模型翻译以及 AI 声音克隆技术，为中文用户提供具有原始 YouTuber 音色的中文配音视频。

## 主要特点
- **AI 语音识别**：有效转换视频中的语音为文字。
- **大型语言模型翻译**：快速且精准地将文本翻译成中文。
- **AI 声音克隆**：生成与原视频配音相似的中文语音。
- **视频处理**：集成的功能实现音视频的同步处理。

## 安装与使用指南
1. **克隆仓库**：
   ```bash
   git clone https://github.com/liuzhao1225/YouDub.git
   ```
2. **安装依赖**：
   进入 `YouDub` 目录并安装所需依赖：
   ```bash
   cd YouDub
   pip install -r requirements.txt
   ```
3. **运行程序**：
   使用以下命令启动主程序：
   ```bash
   python main.py
   ```

## 使用步骤
- 准备需要翻译的视频文件并放置于输入文件夹。
- 指定输出文件夹以接收处理后的视频。
- 系统将自动进行语音识别、翻译、声音克隆和视频处理。
- 
## 技术细节

### AI 语音识别
目前，我们的 AI 语音识别功能是基于 [Whisper](https://github.com/openai/whisper) 实现的。Whisper 是 OpenAI 开发的一款强大的语音识别系统，能够精确地将语音转换为文本。考虑到未来的效率和性能提升，我们计划评估并可能迁移到 [WhisperX](https://github.com/m-bain/whisperX)，这是一个更高效的语音识别系统，旨在进一步提高处理速度和准确度。

### 大型语言模型翻译
我们的翻译功能支持使用 OpenAI API 提供的各种模型，包括官方的 GPT 模型。此外，我们也在探索使用类似 [api-for-open-llm](https://github.com/xusenlinzy/api-for-open-llm) 这样的项目，以便更灵活地整合和利用不同的大型语言模型进行翻译工作。

### AI 声音克隆
声音克隆方面，我们目前使用的是 [Paddle Speech](https://github.com/PaddlePaddle/PaddleSpeech)。虽然 Paddle Speech 提供了高质量的语音合成能力，但目前尚无法在同一句话中同时生成中文和英文。在此之前，我们也考虑过使用 [Coqui AI TTS](https://github.com/coqui-ai/TTS)，它能够进行高效的声音克隆，但同样面临一些限制。

### 视频处理
我们的视频处理功能强调音视频的同步处理，例如确保音频与视频画面的完美对齐，以及生成准确的字幕，从而为用户提供一个无缝的观看体验。

## 贡献指南
欢迎对 `YouDub` 进行贡献。您可以通过 GitHub Issue 或 Pull Request 提交改进建议或报告问题。

## 许可协议
`YouDub` 遵循 Apache License 2.0。使用本工具时，请确保遵守相关的法律和规定，包括版权法、数据保护法和隐私法。未经原始内容创作者和/或版权所有者许可，请勿使用此工具。

## 支持与联系方式
如需帮助或有任何疑问，请通过 [GitHub Issues](https://github.com/liuzhao1225/YouDub/issues) 联系我们。

