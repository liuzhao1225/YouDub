import re
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

class Translator:
    def __init__(self):
        self.client = OpenAI()
        self.system_message = """
        **角色**：翻译专家
        **目标**：将任何给定的文本从任何语言翻译成中文，同时保持其自然性、流畅性和地道性。翻译出的文本应使用优雅和精致的表达。
        **注意**：
        - 确保翻译的准确性。
        - 在表达中保持优雅和精致。
        - 将'text'中的所有数字用中文字符表示。
        - 将'start'和'end'保留为数字。
        - 在翻译文本中保持JSON结构。
        请在```json``中输入您的翻译文本。"""
        
    def translate(self, transcipt):
        print(transcipt)
        print('翻译中...')
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": f"```json\n{transcipt}\n```"},
            ]
        )
        response = response.choices[0].message.content
        try:
            result = re.search(r'```json((.|\n)*?)```', response).group(1).replace("'", '"')
            result = json.loads(result)
        except:
            raise Exception(result)
        return result


        
if __name__ == '__main__':
    translator = Translator()
    translate_from_folder(r'output\Kurzgesagt Channel Trailer', translator)
    
