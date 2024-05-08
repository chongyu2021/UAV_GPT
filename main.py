import re
import os
import io
import sys
from apis import *
from chatgpt import ask,chat_history

code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)


def extract_python_code(content):
    # 提取出chatgpt回复中的代码
    code_blocks = code_block_regex.findall(content)
    if code_blocks:
        full_code = "\n".join(code_blocks)

        if full_code.startswith("python"):
            full_code = full_code[7:]

        return full_code
    else:
        return None


def execute_instruction(instruction):
    # 使用 io.StringIO 捕获执行指令时的输出
    stdout_backup = sys.stdout
    sys.stdout = io.StringIO()

    try:
        exec(instruction)
    finally:
        # 将捕获的输出取出并恢复原来的输出流
        output = sys.stdout.getvalue()
        sys.stdout = stdout_backup

    return output
def main():
    print("欢迎使用Airsim Chatbot!请告诉我接下来的指令")
    while True:
        question = input("AirSim> ")

        if question == "!quit" or question == "!exit":
            break

        if question == "!clear":
            os.system("cls")
            continue

        print("等待chatgpt响应中...")
        response,his = ask(question,chat_history)

        print(f"\n{response}\n")

        while "完成" not in response:
            code = extract_python_code(response)
            if code is not None:
                print("正在执行指令中...")
                output = execute_instruction(code)
                print("指令执行完成!\n")
                his.append(
                    {
                        "role": "assistant",
                        "content": output
                    }
                )
                response, his = ask(question, his)


if __name__ == "__main__":
    main()