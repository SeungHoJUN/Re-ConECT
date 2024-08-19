from langchain_upstage import UpstageLayoutAnalysisLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_upstage import ChatUpstage
from typing import List, Dict

def load_documents(pdf_files):
    loaders = []
    docs = []

    for i, pdf_file in enumerate(pdf_files, start=1):
        print(f"Processing file {i}: {pdf_file}")

        loader = UpstageLayoutAnalysisLoader(
            pdf_file, use_ocr=True, output_type="html"
        )
        loaders.append(loader)

        doc = loader.load()
        docs.append(doc)

        print(f"File {i} processed successfully.")

    print("All files have been processed.")
    return docs

def process_documents(docs: List[List], queries: List[str]) -> Dict[int, Dict[str, str]]:
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        chunk_size=1000, chunk_overlap=100, language=Language.HTML
    )

    llm = ChatUpstage()

    prompt_template = PromptTemplate.from_template(
        """
        Please provide most correct answer from the following context.
        ---
        Question: {question}
        ---
        Context: {Context}
        """
    )
    chains = prompt_template | llm | StrOutputParser()

    results = {}
    for i, doc in enumerate(docs):
        splits = text_splitter.split_documents(doc)
        retriever = BM25Retriever.from_documents(splits)

        doc_results = {}
        for query in queries:
            context_docs = retriever.invoke(query)
            context = chains.invoke({"question": query, "Context": context_docs})
            doc_results[query] = context
            print(f"Document {i}, Query: {query}")
            print(context)
            print("---")

        results[i] = doc_results

    return results