import os
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
#加载环境变量
load_dotenv()

# # 使用 AIHubmix
# Settings.llm = OpenAILike(
#     model="glm-4.7-flash-free",
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     api_base="https://aihubmix.com/v1",
#     is_chat_model=True
# )
#deepseek-chat模型配置
Settings.llm = OpenAILike(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_base="https://api.deepseek.com/v1",
    is_chat_model=True,
    context_window=64000
)
#嵌入模型配置
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# markdown导入
docs = SimpleDirectoryReader(input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]).load_data()

# 建立索引
index = VectorStoreIndex.from_documents(docs)

#建立查询引擎
query_engine = index.as_query_engine()

print(query_engine.get_prompts())

print(query_engine.query("文中举了哪些例子?"))