from ragas import EvaluationDataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
import pandas as pd
from chain import VectorDatabase, LLMHandler, QuestionAnsweringChain
import json
import time
import os
from dotenv import load_dotenv

# Tải biến môi trường
load_dotenv(dotenv_path=r"B:\PROJECTS\CHATBOT_EDUCATION\.env.init")
gemini_key = os.getenv("gemini_key")


def create_dataset_eval():
    # Khởi tạo llm, vectorstore và chain
    path_faiss_index=r"B:\PROJECTS\CHATBOT_EDUCATION\data_source\data\faiss_v2"
    llm_handler = LLMHandler(model_name="gemini-1.5-flash", gemini_key=gemini_key)
    vector_store = VectorDatabase(path_faiss_index=path_faiss_index)
    qa_chain = QuestionAnsweringChain(
        llm_handler=llm_handler,
        vector_db=vector_store,
        num_docs=5,
        apply_rewrite=False,
        apply_rerank=False,
        date_impact=0.01
    )

    # Read data from json file
    path = r"B:\PROJECTS\CHATBOT_EDUCATION\data_source\data\raw\outputs_QA.json"
    data = pd.read_json(path)
    data_dict =  data.to_dict(orient='records')
    # print(data_dict)

    dataset = []

    def safe_invoke_with_retry(invoke_func, retries=5, delay=15):
        for i in range(retries):
            try:
                return invoke_func()
            except Exception as e:
                print(f"[Retry {i+1}/{retries}] Lỗi: {e}")
                time.sleep(delay)
        raise RuntimeError("Gọi API thất bại sau nhiều lần thử")

    for pair in data_dict:
        relevant_docs = safe_invoke_with_retry(lambda: qa_chain.retriever.invoke(pair["question"]))
        response, _ = safe_invoke_with_retry(lambda: qa_chain.run(pair["question"]))
        dataset.append(
            {
                "user_input": pair["question"],
                "retrieved_contexts": [rdoc.page_content for rdoc in relevant_docs],
                "response": response,
                "reference": pair["answer"],
            }
        )
        time.sleep(2)

    dataset_eval_file = r"B:\PROJECTS\CHATBOT_EDUCATION\data_source\data\raw\dataset_eval.json"
    with open(dataset_eval_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)


path = r"B:\PROJECTS\CHATBOT_EDUCATION\data_source\data\raw\dataset_eval.json"
data = pd.read_json(path)
dataset =  data.to_dict(orient='records')

evaluation_dataset = EvaluationDataset.from_list(dataset)

llm_handler = LLMHandler(model_name="gemini-1.5-flash", gemini_key=gemini_key)
evaluator_llm = LangchainLLMWrapper(llm_handler.get_llm())

result = evaluate(
    dataset=evaluation_dataset,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
    llm=evaluator_llm,
)

print(result)