from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chains import ConversationChain, LLMChain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory, ConversationSummaryMemory, ConversationBufferMemory
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
import os
import re
from time import sleep
import plotly.graph_objects as go
import json


with open("api_keys.json", "r") as f:
    api_keys = json.load(f)
os.environ['OPENAI_API_KEY'] = api_keys['openai']


def plot_price_propositions(seller_propositions, buyer_propositions):
    # Once the main loop ends, create a Plotly plot
    fig = go.Figure()
    steps = [i for i in range(1, len(seller_propositions) + 1)]
    # Add price data
    fig.add_trace(go.Scatter(x=steps, y=seller_propositions, mode='lines+markers', name='Seller'))
    fig.add_trace(go.Scatter(x=steps, y=buyer_propositions, mode='lines+markers', name='Buyer'))
    # Add labels
    fig.update_layout(title='Buyer and Seller Price Propositions over Negotiation Steps',
                      xaxis_title='Steps', yaxis_title='Price Proposition (PLN)')
    # Show the figure
    fig.show()


seller = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
buyer = ChatOpenAI(model='gpt-4', temperature=0)
summariser = ChatOpenAI(temperature=0)

# read template from file
with open("prompt_templates/seller_prompt_template-basic.txt", "r", encoding='utf-8') as f:
    seller_prompt_template = f.read()

with open("prompt_templates/buyer_prompt_template-basic.txt", "r", encoding='utf-8') as f:
    buyer_prompt_template = f.read()

seller_prompt = PromptTemplate(
    template=seller_prompt_template,
    input_variables=["buyer_message", "chat_history"],
)
buyer_prompt = PromptTemplate(
    template=buyer_prompt_template,
    input_variables=["seller_message", "chat_history"],
)

buffer_mem = ConversationBufferMemory(memory_key="chat_history", ai_prefix="Seller", human_prefix="Buyer")
summary_mem = ConversationSummaryMemory(llm=summariser, memory_key="chat_history", ai_prefix="Seller",
                                        human_prefix="Buyer")
seller_chain = LLMChain(
    llm=seller,
    prompt=seller_prompt,
    memory=buffer_mem,
    verbose=False,
)
buffer_mem = ConversationBufferMemory(memory_key="chat_history", ai_prefix="Buyer", human_prefix="Seller")
summary_mem = ConversationSummaryMemory(llm=summariser, memory_key="chat_history", ai_prefix="Buyer",
                                        human_prefix="Seller")
buyer_chain = LLMChain(
    llm=buyer,
    prompt=buyer_prompt,
    memory=buffer_mem,
    verbose=False,
)


buyer_prices = []
seller_prices = []
negotiation_round = 0

buyer_response = "[hey! which price you can propose for that house?]"
while negotiation_round < 11:
    buyer_message = re.search(r"\[(.*?)\]", buyer_response).group(1)
    seller_response = seller_chain.run(buyer_message)
    print(seller_response)
    seller_message = re.search(r"\[(.*?)\]", seller_response).group(1)
    seller_price_search = re.search("Price proposal: (.*?)PLN", seller_response)

    if seller_price_search is not None:
        seller_price = seller_price_search.group(1)
        #seller_price = seller_price.replace(',', '')
        seller_prices.append(int(seller_price))
    else:
        print('Seller has decided to end negotiations.')
        break

    buyer_response = buyer_chain.run(seller_message)
    print(buyer_response)
    buyer_price_search = re.search("Price proposal: (.*?)PLN", buyer_response)

    if buyer_price_search is not None:
        buyer_price = buyer_price_search.group(1)
        buyer_prices.append(int(buyer_price))
    else:
        print('Buyer has decided to end negotiations.')
        break

    if buyer_prices[-1] == seller_prices[-1]:
        print('Buyer and seller have agreed on a price. Negotiations are complete.')
        break

    negotiation_round += 1

plot_price_propositions(seller_prices, buyer_prices)
