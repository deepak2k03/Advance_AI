#Standard library imports
import os

#Third-party imports
from openai import OpenAI
from dotenv import load_dotenv

#Local Modules
from tools import TOOLS, run_tool
from Utilities.utilities import log


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
        "content":"You are a helpful humble assistant>"
        "Answer general questions (programming, science, advice) from your own knowledge."
        "Use a tool ONLY when the user asks for current time, the weather or the city."
        "Keep answers under 100 words."
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

            result = run_tool(tool_call)
            
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


