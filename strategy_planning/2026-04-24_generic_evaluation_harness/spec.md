# Planning a generic evaluation harness

Now that we have a problem-specific evaluation harness, let's work on making this into a generic evaluation harness.

## Expected end user flow

1. User uploads their .csv file (with columns "id" and "text" and "gold_label").
2. They select their uploaded dataset from a dropdown menu. We want to persist the .csv file on our end first (this helps make sure that we can link any runs to a specific dataset).
3. They select the models that they want to use for classification.
4. They write or paste in the prompt that they want to use.
5. They define the structured output that they want to use. Let's make this a requirement as we get better results at scale when we have this enabled for models that support it.
6. They press "run"
7. They see a preview of their results and then can export the .csv file.

## Things to be built out

### Frontend

On the frontend side, we can make this a simple single-page application. We can even keep this a Python-only stack and use Streamlit. This helps us iterate quickly and validate user need before committing to a larger customized build.

### Backend

For the backend, we can use FastAPI.

### Deployment

We can deploy this in the following way:

- Frontend: Streamlit Community Cloud (free, quick to ship)
- Backend: Railway

## Scoping

### V1

A V1 can just be a Streamlit frontend over the existing moral-outrage evaluator.

### V2

A V2 will be a generic batch text-classification tool, where users can supply their own prompt and structured schema.
