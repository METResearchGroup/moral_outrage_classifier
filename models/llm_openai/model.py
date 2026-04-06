import os
import requests
from dotenv import load_dotenv

from models.base import BaseModel

load_dotenv(override=True)

class OpenAIModel(BaseModel):

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API Key not found. Pass it to the constructor or set OPENAI_API_KEY environment variable.")

    def _run_prompt(self, prompt: str) -> str:
        url = "https://api.openai.com/v1/responses"
        headers = {
            "Authorization": f"Bearer {os.getenv("OPENAI_API_KEY")}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4o-mini",
            "input": [
                {"role": "system", "content": "You are a concise Python tutor."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_output_tokens": 300,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["output"][0]["content"][0]["text"]

    def batch_classify(self, texts: list[str]) -> list[str]:
        self._validate_input(texts)
        return [self._run_prompt(text) for text in texts]
    

# key = os.getenv("OPENAI_API_KEY")
# print(f"Key length: {len(key) if key else 0}")
# print(f"Last 4 characters: {key[-4:] if key else 'None'}")

oam = OpenAIModel()
print(oam.batch_classify(["what is 1 + 1?", "What is the capital of France?"]))    
