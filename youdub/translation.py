import os
import re
import openai
from dotenv import load_dotenv
import time

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')

system_message = \
"""请你扮演科普专家的角色。这是一个为视频配音设计的翻译任务，将各种语言精准而优雅地转化为尽量简短的中文。请在翻译时避免生硬的直译，而是追求自然流畅、贴近原文而又不失文学韵味的表达。在这个过程中，请特别注意维护中文特有的语序和句式结构，使翻译文本既忠于原意又符合中文的表达习惯。
**注意事项**：
- 紧密关注上下文的逻辑关系，确保翻译的连贯性和准确性。
- 遵循中文的语序原则，即定语放在被修饰的名词前，状语放在谓语前，以保持中文的自然语感。
- 鼓励用自己的话重新诠释文本，避免逐字逐句的直译。采用意译而非直译的方式，用你的话语表达原文的精髓。
- 保留专有名词的原文，如人名、地名、机构名等。
- 化学式用中文表示，例如CO2说二氧化碳，H2O说水。
- 长句子可以分成多个短句子，便于观众理解。
- 使用中文字符。
- 严格遵循回答格式。
"""   

caution = """**注意事项**：
- 紧密关注上下文的逻辑关系，确保翻译的连贯性和准确性。
- 遵循中文的语序原则，即定语放在被修饰的名词前，状语放在谓语前，以保持中文的自然语感。
- 鼓励用自己的话重新诠释文本，避免逐字逐句的直译。采用意译而非直译的方式，用你的话语表达原文的精髓。
- 保留专有名词的原文，如人名、地名、机构名等。
- 化学式用中文表示，例如CO2说二氧化碳，H2O说水。
- 长句子可以分成多个短句子，便于观众理解。
- 使用中文字符。
- 严格遵循回答格式。"""

prefix = '中文：'
class Translator:
    def __init__(self):
        self.system_message = system_message
        self.messages = []
        
    def translate(self, transcript):
        print('总结中...')
        retry = 10
        while retry:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": '你是一个科普专家。你的目的是总结文本中的主要科学知识。'}] + [{"role": "user", "content": f"让我们深呼吸，一步一步地思考。你的总结应该是信息丰富且真实的，涵盖主题的最重要方面。概括这个视频的主要科学内容: {''.join(transcript)}。使用中文简单总结。"},], timeout=240)
                summary = response.choices[0].message.content
                print(summary)
                retry = 0
            except Exception as e:
                retry -= 1
                print('总结失败')
                print(e)
                print('重新总结')
                time.sleep(1)
                if retry == 0:
                    raise Exception('总结失败')
        self.fixed_messages = [{'role': 'user', 'content': 'Hello!'}, {
            'role': 'assistant', 'content': f'{prefix}"你好！"'}, {'role': 'user', 'content': 'Animation videos explaining things with optimistic nihilism since 2,013.'}, {
            'role': 'assistant', 'content': f'{prefix}"从2013年开始，我们以乐观的虚无主义制作动画，进行科普。"'}]
        # self.fixed_messages = []
        self.messages = []
        final_result = []
        print('\n翻译中...')
        for sentence in transcript:
            if not sentence:
                continue
            success = False
            retry_message = ''
            
            # print(messages)
            while not success:
                messages = [{"role": "system", "content": summary + '\n' + self.system_message}] + self.fixed_messages + \
                    self.messages[-20:] + [{"role": "user",
                                            "content": f'{caution}{retry_message}请按照回答格式翻译成中文："{sentence}"'},]
                try:
                    response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.5,
                    timeout=60,
                    )
                    response = response.choices[0].message.content
                    matches = re.findall(r'"((.|\n)*?)"', response)
                    if matches:
                        result = matches[-1][0].replace("'", '"').strip()
                        result = re.sub(r'\（[^)]*\）', '', result)
                        result = result.replace('...', '，')
                    else:
                        result = None
                        raise Exception('没有找到相应格式的翻译结果')
                    if result:
                        self.messages.append(
                            {'role': 'user', 'content': f"{sentence}"})
                        self.messages.append(
                            {'role': 'assistant', 'content': f'{prefix}"{result}"'})
                        print(sentence)
                        print(response)
                        print('='*50)
                        final_result.append(result)
                        success = True
                except Exception as e:
                    print(response)
                    print(e)
                    print('翻译失败')
                    retry_message += f'严格遵循回答格式，将翻译结果放入"引号"中，例如，请翻译成中文："Hello!"，{prefix}"你好！"\n'
                    time.sleep(0.5)
                finally:
                    time.sleep(0.5)
        return final_result


        
if __name__ == '__main__':
    import json
    output_folder = r"output\An Antidote to Dissatisfaction"
    with open(os.path.join(output_folder, 'en.json'), 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    transcript = [sentence['text'] for sentence in transcript if sentence['text']]
    # transcript = ["Humans are apes with smartphones, living on a tiny moist rock, which is speeding around a burning sphere a million times bigger than itself.", "But our star is only one in billions in a milky way, which itself is only one in billions of galaxies.", "Everything around us is filled with complexity, but usually we don't notice, because being a human takes up a lot of time.", "So we try to explain the universe and our existence one video at a time.", "What is life? Are there aliens? What happens if you step on a black hole?", "If you want to find out, you should click here and subscribe to the Kurzgesagt In A Nutshell YouTube channel."]
    print(transcript)
    translator = Translator()
    result = translator.translate(transcript)
    print(result)
    # translate_from_folder(r'output\Kurzgesagt Channel Trailer', translator)
    
