#Standard library imports
import os
from types import SimpleNamespace

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
        "role": "system",
        "content": (
            "You are a helpful, humble assistant for Zorvex Technologies employees. "
            "Answer general questions (programming, science, advice) from your own knowledge. "
            "Use get_current_time, get_city_weather and get_user_city only for the current time, the weather, or the user's city. "
            "For ANY question about Zorvex policies, benefits, leave, medical cover, HR, hiring, tech processes, incident reports or travel expenses, "
            "you MUST first call search_knowledge_base. "
            "Answer company questions ONLY from the text that search_knowledge_base returns. Do not use outside knowledge for company facts. "
            "After the answer, cite the source like (leave_policy.txt #3). "
            "If the returned text does not contain the answer, say exactly: I could not find this in the knowledge base. "
            "Keep answers under 100 words."
        ),
    }
)

def stream_chat(history):
    stream = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=history,
        tools=TOOLS,
        stream=True,
    )

    full_response=""
    calls={}
    label_printed=False

    #streaming started
    try:
        for chunk in stream:
            if not chunk.choices:
                continue
            delta=chunk.choices[0].delta

            #1. content straeaming - do we have messages from the LLM
            if delta.content:
                if not label_printed:
                    print("🤖 Assistant: ", end="", flush=True)
                    label_printed=True
                print(delta.content, end="", flush=True)
                full_response+=delta.content

            #2. Tool calls
            for tc in (delta.tool_calls or []):
                obj = calls.setdefault(
                    tc.index,
                    SimpleNamespace(id=None, function=SimpleNamespace(name="", arguments="")),
                )
                if tc.id:
                    obj.id = tc.id
                if tc.function and tc.function.name:
                    obj.function.name += tc.function.name
                if tc.function and tc.function.arguments:
                    obj.function.arguments += tc.function.arguments
    
    except KeyboardInterrupt:
        print("\n {Stopped by you}")
        calls={}

    if label_printed:
        print() #move to the next line aftere assistant's response

    calls_list = []
    for key in sorted(calls):
        calls_list.append(calls[key])

    return full_response, calls_list

#convert our collected tools_calls into dictionary format
def build_tool_calls_payload(tool_calls):
    payload=[]

    for tool_call in tool_calls:
        tool_call_dict={
            "id":tool_call.id,
            "type":"function",
            "function":{
                "name":tool_call.function.name,
                "arguments":tool_call.function.arguments
            }
        }
        payload.append(tool_call_dict)

    return payload

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
        
        full_response, tool_calls = stream_chat(messages)
        
        #variable to store
        # msg=response.choices[0].message

        if not tool_calls:
            break

        log(f"Step {step+1}: model requested {len(tool_calls)} tool call(s)")

        messages.append(
            {
                "role":"assistant",
                "content":"full_response",
                "tool_calls":build_tool_calls_payload(tool_calls)
            }
        )

        for tool_call in tool_calls:

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

