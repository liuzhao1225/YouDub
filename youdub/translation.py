import re
from openai import OpenAI
from dotenv import load_dotenv


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
- 使用中文字符。
"""   
class Translator:
    def __init__(self):
        self.client = OpenAI()
        self.system_message = system_message
        self.messages = []
        
    def translate(self, transcript):
        print('总结中...')
        retry = 5
        while retry:
            try:
                response = self.client.chat.completions.create(
                            model="gpt-4",
                    messages=[{"role": "system", "content": 'You are a summarizer about video transcripts.'}] + [{"role": "user", "content": f"Your summary should be informative and factual,covering the most important aspects of the topic. Summarize  what is this video is about: {' '.join(transcript)}"},]
                    ,timeout=240)
                summary = response.choices[0].message.content
                print(summary)
                retry = 0
            except Exception as e:
                retry -= 1
                print('总结失败')
                print(e)
                print('重新总结')
                if retry == 0:
                    raise Exception('总结失败')
        self.messages = [{"role": "user", "content": "Begin!"}, {'role': 'assistant', 'content': '好的， 我已经明白了你的要求。在面对这样的翻译任务时，我会秉承以下原则进行工作：1. **理解上下文**：首先深入理解原文的上下文，把握每个句子的意图和背后的逻辑关系。这对于保证翻译的准确性和连贯性至关重要。2. **重视语序和句式**：遵循中文的语序原则，确保译文的自然流畅。例如，把定语放在名词前，状语放在动词前，以符合中文的表达习惯。3. **意译优于直译**：我会用自己的话重新诠释文本，力求不仅传达原文的意思，还要保持其文学韵味和表达风格，避免机械的逐字翻译。4. **保留原文专有名词**：对于人名、地名、机构名等专有名词，我会保留其原文，以保证信息的准确传递。5. **简洁为主**：尽可能地精简译文，避免冗长和复杂的句子结构，使之适合视频配音的需求。6. **分句处理**：将长句子分解为多个短句，便于观众理解和吸收。7. **使用中文字符**：所有翻译均使用标准的中文字符。我的目标是提供既忠实原文又符合中文表达习惯的高品质翻译，既准确又优雅。\n翻译结果：\n```开始！```'}]
        final_result = []
        print('\n翻译中...')
        for sentence in transcript:
            if not sentence:
                continue
            success = False
            retry_message = ''
            while not success:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "system", "content": summary + '\n' + self.system_message}] + self.messages[:-10:max(len(
                        self.messages)//10, 1)] + self.messages[-10:] + [{"role": "user", "content": f"说出你的思考过程，然后将优雅的中文翻译放入```这里```。\n{retry_message}\n请翻译：\n```{sentence}```"},],
                    timeout=120,
                )
                response = response.choices[0].message.content
                try:
                    result = re.search(r'```((.|\n)*?)```', response).group(1).replace("'", '"').strip()
                    if result:
                        self.messages.append(
                            {'role': 'user', 'content': sentence})
                        self.messages.append(
                            # {'role': 'assistant', 'content': f'思考过程：结合上下文。\n翻译：\n```{result}```'})
                            {'role': 'assistant', 'content': response})
                        print(sentence)
                        print(response)
                        print('='*50)
                        final_result.append(result)
                        success = True
                except Exception as e:
                    print(response)
                    print(e)
                    print('翻译失败')
                    if retry_message == '':
                        retry_message += self.system_message
                    else:
                        retry_message += '请将您的翻译文本```放入这里```。'
        return final_result


        
if __name__ == '__main__':
    transcript = ["Humans are apes with smartphones, living on a tiny moist rock, which is speeding around a burning sphere a million times bigger than itself.", "But our star is only one in billions in a milky way, which itself is only one in billions of galaxies.", "Everything around us is filled with complexity, but usually we don't notice, because being a human takes up a lot of time.", "So we try to explain the universe and our existence one video at a time.", "What is life? Are there aliens? What happens if you step on a black hole?", "If you want to find out, you should click here and subscribe to the Kurzgesagt In A Nutshell YouTube channel."]
    print(transcript)
    translator = Translator()
    result = translator.translate(transcript)
    print(result)
    # translate_from_folder(r'output\Kurzgesagt Channel Trailer', translator)
    
