import os
from dotenv import load_dotenv
import dspy
import mlflow


load_dotenv()

lm = dspy.LM('anthropic/claude-3-opus-20240229', api_key=os.getenv('ANTHROPIC_API_KEY'))
dspy.configure(lm=lm)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("DSPy")

mlflow.dspy.autolog()

def evaluate_math(expression: str):
    return dspy.PythonInterpreter({}).execute(expression)

def search_wikipedia(query: str):
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
    return [x['text'] for x in results]

react = dspy.ReAct("question -> answer: str", tools=[evaluate_math, search_wikipedia])


def main():
    output = react(question="what is ww1")
    print(output)


if __name__ == "__main__":
    main()
