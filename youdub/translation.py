import re
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()
system_message = \
"""请你扮演翻译家的角色。这是一个为视频配音设计的翻译任务，将各种语言精准而优雅地转化为尽量简短的中文。请在翻译时避免生硬的直译，而是追求自然流畅、贴近原文而又不失文学韵味的表达。在这个过程中，请特别注意维护中文特有的语序和句式结构，使翻译文本既忠于原意又符合中文的表达习惯。
**注意事项**：
- 紧密关注上下文的逻辑关系，确保翻译的连贯性和准确性。
- 遵循中文的语序原则，即定语放在被修饰的名词前，状语放在谓语前，以保持中文的自然语感。
- 鼓励用自己的话重新诠释文本，避免逐字逐句的直译。采用意译而非直译的方式，用你的话语表达原文的精髓。
- 保留专有名词的原文，如人名、地名、机构名等。
- 翻译尽量简短。
- 长句子可以分成多个短句子，便于观众理解。
请将您简短优美的翻译文本放入以下```代码块```中。
"""   
class Translator:
    def __init__(self):
        self.client = OpenAI()
        self.system_message = system_message
        self.messages = []
        
    def translate(self, transcipt):
        print('翻译中...')
        response = self.client.chat.completions.create(
                    model="gpt-4",
            messages=[{"role": "system", "content": 'You are a summarizer about video transcripts.'}] + [{"role": "user", "content": f"Your summary should be informative and factual,covering the most important aspects of the topic. Summarize  what is this video is about: {' '.join(transcipt)}"},]
                )
        summary = response.choices[0].message.content
        print(summary)
        self.messages = [{"role": "user", "content": "Begin!"}, {'role': 'assistant', 'content': '```开始！```'}]
        final_result = []
        
        for sentence in transcipt:
            if not sentence:
                continue
            success = False
            retry_message = ''
            while not success:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "system", "content": summary + '\n' + self.system_message}] + self.messages[:-10:max(len(
                        self.messages)//10, 1)] + self.messages[-10:] + [{"role": "user", "content": f"{system_message}\n```\n{sentence}\n```{retry_message}"},],
                    timeout=120,
                )
                response = response.choices[0].message.content
                try:
                    result = re.search(r'```((.|\n)*?)```', response).group(1).replace("'", '"').strip()
                    if result:
                        self.messages.append({'role': 'user', 'content': response})
                        self.messages.append(
                            {'role': 'assistant', 'content': f'```\n{result}\n```'})
                        print(sentence)
                        print(result)
                        final_result.append(result)
                        success = True
                except:
                    print('翻译失败')
                    print(response)
                    retry_message += '请将您的翻译文本放入以下```代码块```中。'
        return final_result


        
if __name__ == '__main__':
    transcript = """We present VideoReTalking, a new system to edit the faces of a real-world talking head video according to input audio, producing a high-quality and lip-syncing output video even with a different emotion. Our system disentangles this objective into three sequential tasks: face video generation with a canonical expression, audio-driven lip-sync and face enhancement for improving photo-realism. Given a talking-head video, we first modify the expression of each frame according to the same expression template using the expression editing network, resulting in a video with the canonical expression. This video, together with the given audio, is then fed into the lip-sync network to generate a lip-syncing video. Finally, we improve the photo-realism of the synthesized faces through an identity-aware face enhancement network and post-processing. We use learning-based approaches for all three steps and all our modules can be tackled in a sequential pipeline without any user intervention.""".split('.')
    print(transcript)
    translator = Translator()
    result = translator.translate(transcript)
    print(result)
    # translate_from_folder(r'output\Kurzgesagt Channel Trailer', translator)
    
