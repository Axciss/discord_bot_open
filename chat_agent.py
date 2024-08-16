class Agent:
    def __init__(self, client, llm, system: str = "", max_tokens: int=2048) -> None:
        self.client = client
        self.system = system
        self.llm = llm
        self.max_tokens = max_tokens
        self.messages: list = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message=""):
        if message:
            self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = self.client.chat.completions.create(
            model=self.llm, messages=self.messages, max_tokens=self.max_tokens
        )
        return completion.choices[0].message.content