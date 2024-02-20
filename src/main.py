from langchain.chains import RetrievalQA
from langchain_community.document_loaders import SnowflakeLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
import settings as s

#loader = WebBaseLoader("https://en.wikipedia.org/wiki/Tea")
#documents = loader.load()

snowflake_loader = SnowflakeLoader(
    query=QUERY,
    user=s.SNOWFLAKE_USER,
    password=s.SNOWFLAKE_PASS,
    account=s.SNOWFLAKE_ACCOUNT,
    warehouse=s.SNOWFLAKE_WAREHOUSE,
    role=s.SNOWFLAKE_ROLE,
    database=s.SNOWFLAKE_DATABASE,
    schema=s.SNOWFLAKE_SCHEMA,
)
snowflake_documents = snowflake_loader.load()
text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
texts = text_splitter.split_documents(snowflake_documents)

embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_documents(texts, embeddings)

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())

while True:
    query = input("Ask LibraGPT a question...\n")
    print(qa.run(query))
