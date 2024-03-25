from langchain_openai import ChatOpenAI

from simulation import puzzle_simulation, save_results


llm = ChatOpenAI(model='gpt-4-0613', temperature=0.7, max_tokens=32)
question = "You need to guess a deep learning model. Start your first question now."
answer = "Variational Audoencoder"
results = puzzle_simulation(llm, question, answer, max_it=10)

save_results(results)
