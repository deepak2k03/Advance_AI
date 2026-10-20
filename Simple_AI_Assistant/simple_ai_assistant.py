from openai import OpenAI
from dotenv import load_dotenv
from tools import TOOLS, FUNCTION_NAME_TO_FUNCTION_MAP
from Utilities.utilities import log
import json
import os

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

messages = []

#to stop infinite tool calling
MAX_STEPS=5

#setting up nature of responses of LLM via system
messages.append(
    {
        "role":"system",
        "content":"Always act like a humble agent and respond always under 20 words"
    }
)


while True: #infinite chat implementation

    user_input = input("\n🧑 Deepak: ");

    if user_input.lower() in ["exit", "quit", "bye"]:
        break

    #start preserving conversation
    messages.append(
        {
            "role":"user",
            "content":user_input
        }
    )


    for step in range(MAX_STEPS): #check for chained tool calls
        
        response = client.chat.completions.create(
            model=os.getenv("MODEL"),
            messages=messages,
            tools=TOOLS
        )
        
        #variable to store
        msg=response.choices[0].message

        if not msg.tool_calls:
            break

        log(f"Step {step+1}: model requested {len(msg.tool_calls)} tool call(s)")

        messages.append(msg)

        for tool_call in msg.tool_calls:
            func_name=tool_call.function.name
            func_args={}

            try:
                func_args=json.loads(tool_call.function.arguments or "{}")
                log(f"🔧 {func_name}({func_args})")
                fn = FUNCTION_NAME_TO_FUNCTION_MAP[func_name]
                result = fn(**func_args)
            except Exception as e:
                result=f"Erro while running {func_name} : {e}"

            log(f"   ⤷ {result}")

            messages.append({
                "role":"tool",
                "tool_call_id":tool_call.id,
                "content":str(result)

            })
    else : 
        # else block is for model never gave answer
        fallback = "Sorry! I couldn't finish this within the step limit."
        messages.append({
            "role":"assistant",
            "content":fallback
        })

        print("🤖 Assistant: " + fallback)
        continue

    messages.append(
        {
            "role":"assistant",
            "content":msg.content
        }
    )

    print("🤖 Assistant: " + (msg.content or ""))


