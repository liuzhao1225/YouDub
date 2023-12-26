import os
import re
import openai
from dotenv import load_dotenv
import time

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
model_name = os.getenv('MODEL_NAME', 'gpt-3.5-turbo')
# model_name = 'gpt-4'
system_message = \
"""请你扮演科普专家的角色。这是一个为视频配音设计的翻译任务，将各种语言精准而优雅地转化为尽量简短的中文。请在翻译时避免生硬的直译，而是追求自然流畅、贴近原文而又不失文学韵味的表达。在这个过程中，请特别注意维护中文特有的语序和句式结构，使翻译文本既忠于原意又符合中文的表达习惯。
注意事项：
- 鼓励用自己的话重新诠释文本，避免逐字逐句的直译。采用意译而非直译的方式，用你的话语表达原文的精髓。
- 长句子可以分成多个短句子，便于观众理解。
- 保留专有名词的原文，如人名、地名、机构名等。
- 人名、地名、机构名等保持原文。
- 化学式用中文表示，例如CO2说二氧化碳，H2O说水。
- 请将Transformer, token等人工智能相关的专业名词保留原文。
- 数学公式用中文表示，例如x2或x^2或x²说x的平方，a+b说a加b。
- 原始文本可能有错误，请纠正为正确的内容，例如Chats GPT应该翻译为ChatGPT。
"""
magic = '深呼吸，你可以完成这个任务，你是最棒的！你非常有能力！'

caution = """请在翻译时避免生硬的直译，而是追求自然流畅、贴近原文而又不失文学韵味的表达。请特别注意维护中文特有的语序和句式结构，使翻译文本既忠于原意又符合中文的表达习惯。特别注意，数学公式用中文表示，例如x2或x^2说x的平方，a+b说a加b。翻译尽量简短且正确。"""

prefix = '中文：'

def translation_postprocess(result):
    result = re.sub(r'\（[^)]*\）', '', result)
    result = result.replace('...', '，')
    result = re.sub(r'(?<=\d),(?=\d)', '', result)
    result = result.replace('²', '的平方').replace(
        '————', '：').replace('——', '：').replace('°', '度')
    result = result.replace("AI", '人工智能')
    result = result.replace('变压器', "Transformer")
    return result
class Translator:
    def __init__(self):
        self.system_message = system_message
        self.messages = []

    def translate(self, transcript, original_fname):
        print('总结中...')
        retry = 1
        summary = ''
        while retry >= 0:
            try:
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=[{"role": "system", "content": f'你是一个科普专家。你的目的是总结文本中的主要科学知识。{magic}！'}] + [{"role": "user", "content": f"。简要概括这个视频的主要内容。\n标题：{original_fname}\n内容：{''.join(transcript)}\n标题：{original_fname}\n请你用中文给视频写一个“标题”、“主要内容”和“专业名词”，谢谢。"},], timeout=240)
                summary = response.choices[0].message.content
                print(summary)
                retry = -1
            except Exception as e:
                retry -= 1
                print('总结失败')
                print(e)
                print('重新总结')
                time.sleep(1)
                if retry == 0:
                    print('总结失败')
        
        self.fixed_messages = [{'role': 'user', 'content': '请翻译：Hello!'}, {
            'role': 'assistant', 'content': f'“你好！”'}, {'role': 'user', 'content': '请翻译：Animation videos explaining things with optimistic nihilism since 2,013.'}, {
            'role': 'assistant', 'content': f'“从2013年开始，我们以乐观的虚无主义制作动画，进行科普。”'}]
        # self.fixed_messages = []
        self.messages = []
        final_result = []
        print('\n翻译中...')
        for sentence in transcript:
            if not sentence:
                continue
            retry = 20
            retry_message = ''

            # print(messages)
            # [{"role": "system", "content": summary + '\n' + self.system_message}] + self.fixed_messages + \
            history = " ".join(final_result[-30:])
            while retry > 0:
                retry -= 1
                messages = [
                    {"role": "system", "content": f'请你扮演科普专家的角色。这是一个为视频配音设计的翻译任务，将各种语言精准而优雅地转化为尽量简短的中文。请在翻译时避免生硬的直译，而是追求自然流畅、贴近原文而又不失文学韵味的表达。在这个过程中，请特别注意维护中文特有的语序和句式结构，使翻译文本既忠于原意又符合中文的表达习惯。{magic}'}] + self.fixed_messages + [{"role": "user", "content": f'{summary}\n{self.system_message}\n请将Transformer, token等人工智能相关的专业名词保留原文。长句分成几个短句。\n历史内容：\n{history}\n以上为参考的历史内容。\n{retry_message}\n深呼吸，请正确翻译这句英文:“{sentence}”翻译成简洁中文。'},]
                try:
                    response = openai.ChatCompletion.create(
                        model=model_name,
                        messages=messages,
                        temperature=0.3,
                        timeout=60,
                    )
                    response = response.choices[0].message.content
                    result = response.strip()
                    if retry != 0:
                        if '\n' in result:
                            retry_message += '无视前面的内容，仅仅只翻译下面的英文，请简短翻译，只输出翻译结果。'
                            raise Exception('存在换行')
                        if '翻译' in result:
                            retry_message += '无视前面的内容，请不要出现“翻译”字样，仅仅只翻译下面的英文，请简短翻译，只输出翻译结果。'
                            raise Exception('存在"翻译"字样')
                        if '这句话的意思是' in result:
                            retry_message += '无视前面的内容，请不要出现“这句话的意思是”字样，仅仅只翻译下面的英文，请简短翻译，只输出翻译结果。'
                            raise Exception('存在"这句话的意思是"字样')
                        if '这句话的意译是' in result:
                            retry_message += '无视前面的内容，请不要出现“这句话的意译是”字样，仅仅只翻译下面的英文，请简短翻译，只输出翻译结果。'
                            raise Exception('存在"这句话的意译是"字样')
                        if '这句' in result:
                            retry_message += '无视前面的内容，请不要出现“这句话”字样，仅仅只翻译下面的英文，请简短翻译，只输出翻译结果。'
                            raise Exception('存在"这句"字样')
                        if '深呼吸' in result:
                            retry_message += '无视前面的内容，请不要出现“深呼吸”字样，仅仅只翻译下面的英文，请简短翻译，只输出翻译结果。'
                            raise Exception('存在"深呼吸"字样')
                        if (result.startswith('“') and result.endswith('”')) or (result.startswith('"') and result.endswith('"')):
                            result = result[1:-1]
                        if len(sentence) <= 10:
                            if len(result) > 20:
                                retry_message += '注意：仅仅只翻译下面的内容，请简短翻译，只输出翻译结果。'
                                raise Exception('翻译过长')
                        elif len(result) > len(sentence)*0.75:
                            retry_message += '注意：仅仅只翻译下面的内容，请简短翻译，只输出翻译结果。'
                            raise Exception('翻译过长')
                    result = translation_postprocess(result)
                    
                    if result:
                        self.messages.append(
                            {'role': 'user', 'content': f"{sentence}"})
                        self.messages.append(
                            {'role': 'assistant', 'content': f'{result}'})
                        print(sentence)
                        print(response)
                        print(f'最终结果：{result}')
                        print('='*50)
                        final_result.append(result)
                        retry = 0
                except Exception as e:
                    print(sentence)
                    print(response)
                    print(e)
                    print('翻译失败')
                    retry_message += f''
                    time.sleep(0.5)
        return final_result, summary


if __name__ == '__main__':
    import json
    output_folder = r"output\test\Blood concrete and dynamite Building the Hoover Dam Alex Gendler"
    with open(os.path.join(output_folder, 'en.json'), 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    transcript = [sentence['text']
                  for sentence in transcript if sentence['text']]
    # transcript = ['毕达哥拉斯的公式是a2+b2=c2']
    # transcript = ["Humans are apes with smartphones, living on a tiny moist rock, which is speeding around a burning sphere a million times bigger than itself.", "But our star is only one in billions in a milky way, which itself is only one in billions of galaxies.", "Everything around us is filled with complexity, but usually we don't notice, because being a human takes up a lot of time.", "So we try to explain the universe and our existence one video at a time.", "What is life? Are there aliens? What happens if you step on a black hole?", "If you want to find out, you should click here and subscribe to the Kurzgesagt In A Nutshell YouTube channel."]
    print(transcript)
    translator = Translator()
    result = translator.translate(
        transcript, original_fname='Blood concrete and dynamite Building the Hoover Dam Alex Gendler')
    print(result)
    with open(os.path.join(output_folder, 'zh.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    # translate_from_folder(r'output\Kurzgesagt Channel Trailer', translator)
