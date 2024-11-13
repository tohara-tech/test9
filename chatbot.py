import os

import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain import PromptTemplate
from langchain.schema import AIMessage, HumanMessage
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()


def initialize_vector_store() -> Chroma:
    """VectorStoreの初期化."""
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("DEPLOYMENT_EMBEDDINGS_NAME"),
        openai_api_version="2023-05-15",
    )

    vector_store_path = "/app/resources/note.db"
    if os.path.exists(vector_store_path):
        vector_store = Chroma(embedding_function=embeddings, persist_directory=vector_store_path)
    else:
        loader = TextLoader("/app/resources/note.txt",encoding="utf-8_sig")
        docs = loader.load()

        #text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        text_splitter = CharacterTextSplitter(separator = "\\", chunk_size = 50, chunk_overlap = 20)
        splits = text_splitter.split_documents(docs)
        print(splits)
        vector_store = Chroma.from_documents(
            documents=splits, embedding=embeddings, persist_directory=vector_store_path
        )

    return vector_store


def initialize_retriever() -> VectorStoreRetriever:
    """Retrieverの初期化."""
    vector_store = initialize_vector_store()
    #return vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.6})
    return vector_store.as_retriever(search_kwargs={"k": 6})
    #return vector_store.as_retriever()


def initialize_chain() -> RunnableSequence:
    """Langchainの初期化."""
    #prompt = hub.pull("rlm/rag-prompt")
    template = """
        あなたは質問応答のアシスタントです。
        質問に答えるために、contextの部分を使用してください。
        答えがわからない場合は、contextは無視して回答してください。
        Question: {question} 
        Context: {context} 
        Answer:
    """
    prompt = PromptTemplate(template=template)

    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_version="2024-08-01-preview",
        deployment_name=os.getenv("DEPLOYMENT_GPT_NAME"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        openai_api_type="azure",
    )
    retriever = initialize_retriever()
    chain = (
        {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
    )
    return chain


def main() -> None:
    """ChatGPTを使ったチャットボットのメイン関数."""
    chain = initialize_chain()

    # ページの設定
    st.set_page_config(page_title="RAG ChatGPT")
    st.header("RAG ChatGPT")

    # チャット履歴の初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ユーザーの入力を監視
    if user_input := st.chat_input("聞きたいことを入力してね！"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("GPT is typing ..."):
            response = chain.invoke(user_input)
            print(response)
        st.session_state.messages.append(AIMessage(content=response.content))

    # チャット履歴の表示
    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        else:
            st.write(f"System message: {message.content}")


if __name__ == "__main__":
    main()
