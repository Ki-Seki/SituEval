import json
import os
from datetime import datetime

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


TESTER_SYSTEM = "Question: {question}\nAnswer: {answer}"
TESTER_AI = "Ask me a question, and I will respond with only '[YES]' or '[NO]', without providing any additional information. If you guessed finally, I will responed with '[GUESSED]'. Can you deduce the answer to the question through this method? If you're ready, please ask your first question."

TESTEE_SYSTEM = "You are participating in a game where your task is to figure out the answer to a question. You can only ask yes-or-no questions, and the user will respond with only '[YES]' or '[NO]'."
TESTEE_HUMAN = "{question}"

GOT_IT = "[GUESSED]"


def puzzle_simulation(llm: BaseLanguageModel, question: str, answer: str, max_it: int) -> dict:

    tester_messages = [
        SystemMessage(content=TESTER_SYSTEM.format(question=question, answer=answer)),
        AIMessage(content=TESTER_AI)
    ]

    testee_messages = [
        SystemMessage(content=TESTEE_SYSTEM),
        HumanMessage(content=TESTEE_HUMAN.format(question=question)),
    ]

    print(f'{question=}, {answer=}')
    it = 0
    while True:
        it += 1
        testee_question = llm.invoke(testee_messages)
        testee_messages.append(testee_question)
        tester_messages.append(HumanMessage(content=testee_question.content))

        tester_response = llm.invoke(tester_messages)
        tester_messages.append(tester_response)
        testee_messages.append(HumanMessage(content=tester_response.content))

        print(f'testee={testee_question.content}')
        print(f'tester={tester_response.content}')

        if GOT_IT in tester_response.content:
            status_code = 0
            break

        if it > max_it:
            status_code = -1
            break

    return {
        'timestamp': datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        'status_code': status_code,
        'iterations': it,
        'max_it': max_it,
        'llm': llm.dict(),
        'question': question,
        'answer': answer,
        'tester_messages': [dict(msg.to_json()) for msg in tester_messages],
        'testee_messages': [dict(msg.to_json()) for msg in testee_messages],
    }


def save_results(results: dict, dir: str = 'output'):
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(f'{dir}/{results["timestamp"]}.json', 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
