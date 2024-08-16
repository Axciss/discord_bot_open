import discord
from discord.ext import commands
import os
from groq import Groq
from new_agents import Agent

import argparse
import populate_database
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function
from dotenv import load_dotenv
load_dotenv()

BOT_TOKEN = os.getenv('DISCORD_TOKEN')
HF_TOKEN = os.getenv('HF_TOKEN')

intents = discord.Intents.default()
intents.message_content = True  # Enable message content intent
bot = commands.Bot(command_prefix='!', intents=intents)

# Set the GROQ API key
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Set the channel ID where the bot will respond to messages
channel_id = [0000000000000000000,  # channel ID
              0000000000000000000,  # channel ID
              0000000000000000000   # channel ID
              ]  # channel ID

# define the model
groq_model = "llama-3.1-70b-versatile"
embed_model = "sentence-transformers/all-MiniLM-L6-v2"
# define the system prompt
system_prompt = """
You are a helpful assistant that can answer any question and available to use tools. And you will answer the query in user query's language.
"""

agent = Agent(client=client, system=system_prompt, model=groq_model)

previous_message = None
previous_response = None

# --------------------------------------------------------------------------------
# populate RAG database every time while starting
populate_database.populate()

# --------------------------------------------------------------------------------

@bot.event
async def on_message(message):
    global previous_message
    global previous_response
    print("Received message:", message.content)
    chunk_size = 2000
    if message.channel.id in channel_id and not message.author.bot:
        try:
            ai_msg = agent.run_conversation(message.content, previous_message, previous_response)
            response_text = ai_msg
            # send the md file if the messege is too long (>2000 discord limit)
            if len(response_text) > chunk_size:
                with open('message.md', 'w', encoding='utf-8') as f:
                    f.write(f"# long message\n\n{response_text}")
                await message.channel.send(f"{message.author.mention} The message is too long.", file=discord.File('message.md'))
                os.remove('message.md')
            else:
                await message.channel.send(f"{message.author.mention} {response_text}")
        except Exception as e:
            print("Error invoking GROQ API:", e)
        
        previous_message = message.content
        previous_response = response_text

bot.run(BOT_TOKEN)