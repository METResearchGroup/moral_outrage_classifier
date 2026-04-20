MORAL_OUTRAGE_PROMPT = """

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

You must return an output in the following JSON format: {"label": 0} or {"label": 1}.

Return JSON only:
- If the post describes, reports, or implies moral outrage: {"label": 1}
- Otherwise: {"label": 0}

Do not return plain text, explanations, or extra keys.

Example:

Post: "How dare they let children go hungry while executives collect bonuses."
Answer: {"label": 1}

Post: "I'm so glad I'm not a parent. It's so hard to raise kids these days."
Answer: {"label": 0}

Post: "Why is it always the poor who have to pay for the mistakes of the rich?"
Answer: {"label": 1}

Post: "The weather was nice today and I took a long walk."
Answer: {"label": 0}

Post: {{input}}
Answer:

"""
