from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    return PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

def retrieval_qa_chain(llm, prompt, db):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

def load_llm():
    # Load the locally downloaded model
    try:
        llm = CTransformers(
            model="llama-2-7b-chat.ggmlv3.q8_0.bin",
            model_type="llama",
            max_new_tokens=356,
            temperature=0.4
        )
        return llm
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    return retrieval_qa_chain(llm, qa_prompt, db)

def final_result(query):
    try:
        qa = qa_bot()
        response = qa({'query': query})
        return response
    except Exception as e:
        print(f"Error processing query: {e}")
        return {"result": "An error occurred while processing the query."}

@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to ONGC Bot. What is your query?"
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True

    try:
        res = await chain.acall(message.content, callbacks=[cb])
        answer = res["result"]
        sources = res.get("source_documents", [])

        if sources:
            unique_sources = list(set([str(source.metadata.get('source', '')) for source in sources]))
            answer += "\nSources:\n" + "\n".join(unique_sources)
        else:
            answer += "\nNo sources found."

        await cl.Message(content=answer).send()
    except Exception as e:
        print(f"Error processing message: {e}")
        await cl.Message(content="An error occurred while processing the message.").send()
