import os
from uuid import uuid4
from googleapiclient import discovery
from dotenv import load_dotenv

from lib.timestamp_utils import get_current_timestamp
from schemas.responses import MoralOutrage

load_dotenv()

class PerspectiveAPIModel:
    def __init__(self, api_key: str | None = None) -> None:
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

    def batch_classify(self, texts: list[str]) -> list[MoralOutrage]:
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

        timestamp = get_current_timestamp()
        try:
            res = [
                MoralOutrage(
                    text_id=str(uuid4()),
                    text=text,
                    moral_outrage_score=resp['attributeScores']['MORAL_OUTRAGE_EXPERIMENTAL']['summaryScore']['value'],
                    label_timestamp=timestamp
                )
                for (text, resp) in (zip(texts, response, strict=True))
            ]
        except KeyError as e:
            print(f"Error processing response: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise
    
        return res 

