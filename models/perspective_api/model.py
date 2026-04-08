import os
from googleapiclient import discovery
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

from lib.timestamp_utils import get_current_timestamp
from models.base import BaseModel
from schemas.responses import MoralOutrage

PROB_LABEL_THRESHOLD=0.7

class PerspectiveAPIModel(BaseModel):
    def __init__(self, api_key: str | None = None) -> None:
        load_dotenv()
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

    def batch_classify(self, texts: list[str], text_ids: list[str] | None = None) -> list[MoralOutrage | None]:
        self._validate_input(texts)

        analyze_requests = [
            {
                'comment': { 'text': text },
                'requestedAttributes': {'MORAL_OUTRAGE_EXPERIMENTAL': {}}
            }
            for text in texts
        ]

        responses = []
        for text, analyze_request in zip(texts, analyze_requests, strict=True):                                                                                      
            try:                                                                                                                                                     
                response = self.client.comments().analyze(body=analyze_request).execute()
                responses.append(response)                                                                                                                           
            except HttpError as e:
                if "LANGUAGE_NOT_SUPPORTED_BY_ATTRIBUTE" in str(e):
                    responses.append(None)
                else:
                    raise RuntimeError(f"API error for text '{text[:50]}': {e}") from e  

        timestamp = get_current_timestamp()
        try:
            res = [
                MoralOutrage(
                    text_id=text_id,
                    text=text,
                    moral_outrage_score=resp['attributeScores']['MORAL_OUTRAGE_EXPERIMENTAL']['summaryScore']['value'],
                    label_timestamp=timestamp
                ) if resp is not None else None
                for (text, text_id, resp) in (zip(texts, text_ids, responses, strict=True))
            ]
        except KeyError as e:
            print(f"Error processing response: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise
    
        return res 

