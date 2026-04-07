import os
from dotenv import load_dotenv
from openai import OpenAI
from uuid import uuid4

from models.base import BaseModel
from schemas.responses import MoralOutrage
from lib.timestamp_utils import get_current_timestamp

load_dotenv(override=True)

class OpenAIModel(BaseModel):

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API Key not found. Pass it to the constructor or set OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=self.api_key)

    def _run_prompt(self, text: str) -> str:
        prompt = f"""
        You are a helpful assistant. Your job is to analyze a single social media post and answer a binary classification question.

        The key definition of moral outrage included the following three components: a person can be viewed as expressing moral outrage if (1) they have feelings in response to a perceived violation of their personal morals, (2) their feelings are comprised of emotions such as anger, disgust and contempt, and (3) the feelings are associated with specific reactions including blaming people/events/things, holding them responsible, or wanting to punish them.

        Moral outrage is a emotional response associated with a host of reactions. Most of us just “know” what outrage is based on our experience. But to be more precise, we can say someone is morally outraged when:

        (1) They have feelings in response to a violation of their morals. For instance, Bob thinks that abortion is morally wrong, and when his city passed a law that legalized abortion he became very upset. Since the law upset him because it violated
        his moral views about abortion, the feeling he felt might be moral outrage.

        (2) Their feelings are comprised of emotions such as anger, disgust and contempt. For instance, Bob was upset but specifically he was angered and disgusted at the fact that abortion would become legal in his city. These emotions made him feel very negative and also “worked up”.

        (3) They are associated with specific reactions: blaming people/events/things, holding them responsible, or wanting to punish them. For instance, Bob blamed all the non-religious voters in his city for allowing the law to
        pass. He also felt like the city council members should be held responsible for allowing
        the law to actually become reality. For all these reasons, we can say Bob is morally outraged about the legalizing of abortion
        in his city.

        - If you judge that the post describes, reports, or implies moral outrage, respond with: "1"
        - If the post is unrelated or does not describe moral outrage, respond with: "0"

        Only output your label. ONLY output 0 or 1.

        Post: {text}
        Answer:
        """

        input = [
            {"role": "user", "content": prompt}
        ]
        response = self.client.responses.create(
            model="gpt-5-nano",
            input=input,
            max_output_tokens=300,
        )
        output = response.output[1] # response.output[0] for gpt-5.4
        answer = output.content[0].text
        return answer


    def batch_classify(self, texts: list[str]) -> list[MoralOutrage]:
        self._validate_input(texts)
        inference_results = [self._run_prompt(text) for text in texts]
        moral_outrage_list = []
        timestamp = get_current_timestamp()
        try:
            for inference, text in zip(inference_results, texts, strict=True):
                moral_outrage_list.append(MoralOutrage(
                    text_id=str(uuid4()),
                    text=text,
                    moral_outrage_score=int(inference),
                    label_timestamp=timestamp,
                ))   
        except:
            raise Exception("At least one of the responses did not return either 0 or 1")
    
        return moral_outrage_list
