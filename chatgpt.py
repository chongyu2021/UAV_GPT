import requests
import json

# prompt的路径
sys_prompt_path = "prompts/basic.txt"
with open(sys_prompt_path, "r", encoding="utf-8") as f:
    sys_prompt = f.read()

# 自定义chat History
chat_history = [
    {
        "role": "user",
        "content": sys_prompt
    },
    {
        "role": "assistant",
        "content": "好的，我将作为你的助手帮助你使用AirSim仿真器进行无人机操作。请告诉我你希望无人机执行的第一个任务，我会为你提供相应的Python代码并解释其作用。"
    },
    {
        "role": "user",
        "content": "起飞并移动到(10,10,5)"
    },
    {
        "role": "assistant",
        "content": """```python
    # 使无人机起飞  
    takeoff()  
    ```
    """
    },
    {
        "role": "user",
        "content": "起飞成功！"
    },
    {
        "role": "assistant",
        "content": """```python
        # 获取当前位置  
        current_position = get_position()  
        ```
        """
    },
    {
        "role": "user",
        "content": "当前位置是(0,0,0)"
    },
    {
        "role": "assistant",
        "content": """```python
        # 移动到目标地点 
        move_to_position(10,10,5)
        ```
        """
    },
    {
        "role": "user",
        "content": "成功移动到(10,10,5)"
    },
    {
        "role": "assistant",
        "content": """
        任务完成！
        """
    }
]


def ask(prompt,history):
    url = "https://apikeyplus.com/v1/chat/completions"

    headers = {
        'Content-Type': 'application/json',
        # 填写 OpenKEY 生成的令牌/KEY，注意前面的 Bearer 要保留，并且和 KEY 中间有一个空格。
        'Authorization': 'Bearer sk-vintOWhjGERJPMNZD3F5C417E8C949F991224eA2E70fE236'
    }

    history.append(
        {
            "role": "user",
            "content": prompt,
        }
    )
    data = {
        "model": "gpt-3.5-turbo",
        "messages": history
    }

    response = requests.post(url, headers=headers, json=data)

    text = json.loads(response.text)['choices'][0]['message']['content']
    history.append(
        {
            "role": "assistant",
            "content": text
        }
    )
    return text, history


if __name__ == "__main__":
    print(ask("起飞并降落"))
