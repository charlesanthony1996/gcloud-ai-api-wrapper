# base class init here
class BaseLLMClient:
    def __init__(self, api_key):
        self.api_key = api_key

    def create_completion(self, prompt):
        pass
    def create_completionCS(self, prompt):
        pass

class OpenAIClient(BaseLLMClient):
    def create_completion(self, prompt):
        client = OpenAI(api_key=self.api_key)
        return client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Detect hate speech and respond 'Yes' or if no hate speech respond 'No hate speech detected.'"},
                {"role": "user", "content": prompt}
            ],
        )
    def create_completionCS(self, prompt):
        client = OpenAI(api_key=self.api_key)
        return client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Generate counter speech"},
                {"role": "user", "content": prompt}
            ],
        )
    def explain_HS(self, prompt):
        client = Groq(api_key=self.api_key)
        return client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a model designed to explain why a given text is hate speech. Please provide an explanation."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192",
            temperature=0.5,
            max_tokens=150,
            top_p=1,
            stop=None,
            stream=False,
        )

class GroqClient(BaseLLMClient):
    def create_completion(self, prompt):
        client = Groq(api_key=self.api_key)
        return client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Detect hate speech and respond 'Yes' or if no hate speech respond 'No hate speech detected.'"},
                {"role": "user", "content": prompt}
            ],
            model="mixtral-8x7b-32768",
            temperature=0.5,
            max_tokens=150,
            top_p=1,
            stop=None,
            stream=False,
        )
    
    def create_completionCS(self, prompt):
        client = Groq(api_key=self.api_key)
        return client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Generate counter speech"},
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192",
            temperature=0.5,
            max_tokens=150,
            top_p=1,
            stop=None,
            stream=False,
        )
    
    def explain_HS(self, prompt):
        client = Groq(api_key=self.api_key)
        return client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a model designed to explain why a given text is hate speech. Please provide an explanation."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192",
            temperature=0.5,
            max_tokens=150,
            top_p=1,
            stop=None,
            stream=False,
        )