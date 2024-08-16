import json
from tools import get_time, RAG_mixture_of_agents


class Agent:
    def __init__(self, client, model, system: str = "") -> None:
        self.client = client
        self.model = model
        self.system=system

    def run_conversation(self, user_prompt, pre_prompt: str=None, pre_response: str=None):
        if pre_prompt is not None:
            messages = [
                {
                    "role": "system",
                    "content": self.system
                },
                {
                    "role": "user",
                    "content": pre_prompt,
                },
                {
                    "role": "assistant",
                    "content": pre_response,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": self.system
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_utc_time",
                    "description": "Get the current UTC time.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "location"},
                        },
                        "required": ["location"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "RAG_mixture_of_agents",
                    "description": "Search informations about mixture of agents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string", "description": "user query"},
                        },
                        "required": ["message"],
                    },
                },
            },
        ]

        while True:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=4096
            )

            response_message = response.choices[0].message

            print("----------------------------------------")
            print(response_message.content)
            print("----------------------------------------")
            messages.append(response_message)

            if not response_message.tool_calls:
                return response_message.content

            available_functions = {
                "get_utc_time": get_time,
                "RAG_mixture_of_agents":RAG_mixture_of_agents,
            }

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(**function_args)
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )