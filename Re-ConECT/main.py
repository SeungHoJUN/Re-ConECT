from Diagnosis_process import Diagnosis_Process
from Rehabilitation_assessment import rehabilitation_assessment_workflow
from RAG import load_documents
from langchain_upstage import ChatUpstage

def check_diagnosis(llm, docs):
    has_diagnosis = input("Do you have a diagnostic assessment? (yes/no): ").lower()
    
    if has_diagnosis == "yes":
        diagnosis_id = input("Please enter the diagnostic assessment ID: ")
        file_path = f'/content/drive/MyDrive/240814_Llama_RAG/{diagnosis_id}.csv'
        rehabilitation_assessment_workflow(llm, docs, file_path)
    elif has_diagnosis == "no":
        Diagnosis_Process()
    else:
        print("Invalid input. Please answer 'yes' or 'no'.")
        check_diagnosis(llm, docs)  # Recursively call the function to get correct input

if __name__ == "__main__":
    # Initialize the language model and document list
    llm = ChatUpstage(temperature=0)
    docs = load_documents('path/data')
    check_diagnosis(llm, docs)