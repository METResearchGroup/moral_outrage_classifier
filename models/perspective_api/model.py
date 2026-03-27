import os
import json
from googleapiclient import discovery
from dotenv import load_dotenv

load_dotenv()

class PerspectiveAPIModel:
    def __init__(self, api_key: str = None) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError("Perspective API Key not found. Pass it to the constructor or set GOOGLE_API_KEY environment variable.")
        
        self.client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=self.api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
            )

    def batch_classify(self, texts: list[str]) -> list[dict]:
        analyze_requests = [
            {
                'comment': { 'text': text },
                'requestedAttributes': {'MORAL_OUTRAGE_EXPERIMENTAL': {}}
            }
            for text in texts
        ]

        response = [
            self.client.comments().analyze(body=analyze_request).execute()
            for analyze_request in analyze_requests
        ]

        print(json.dumps(response, indent=2))
        return response 


pam = PerspectiveAPIModel()
pam.batch_classify(["friendly greetings from python", "you are a terrible person"])


# analyze_request = {
#   'comment': { 'text': 'friendly greetings from python' },
#   'requestedAttributes': {'TOXICITY': {}}
# }

# response = client.comments().analyze(body=analyze_request).execute()
# print(json.dumps(response, indent=2))