import asyncio
import base64
from datetime import datetime as dt
import re

import aiohttp
from aiohttp import ClientTimeout

import nltk
from bs4 import BeautifulSoup
import logging
from os import environ as env
import requests

import discord
from dotenv import load_dotenv
from litellm import acompletion

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# Add diagnostic logging for environment variables
logging.info(f"LLM: {env['LLM']}")
logging.info(f"DISCORD_CLIENT_ID: {env['DISCORD_CLIENT_ID']}")
# Add logging for other environment variables as needed

LOCAL_LLM: bool = env["LLM"].startswith("local/")
VISION_LLM: bool = any(x in env["LLM"] for x in ("claude-3", "llava", "vision"))
LLM_SUPPORTS_MESSAGE_NAME: bool = True
#LLM_SUPPORTS_MESSAGE_NAME: bool = any(env["LLM"].startswith(x) for x in ("gpt", "openai/gpt")) and "gpt-4-vision-preview" not in env["LLM"]

BOT_ROLE_ID = 1225893397810643049  # Update with your bot's role ID
ALLOWED_CHANNEL_TYPES = (discord.ChannelType.text, discord.ChannelType.public_thread, discord.ChannelType.private_thread, discord.ChannelType.private)
ALLOWED_CHANNEL_IDS = tuple(int(id) for id in env["ALLOWED_CHANNEL_IDS"].split(",") if id)
ALLOWED_ROLE_IDS = tuple(int(id) for id in env["ALLOWED_ROLE_IDS"].split(",") if id)
MAX_IMAGES = int(env["MAX_IMAGES"]) if VISION_LLM else 0
MAX_MESSAGES = int(env["MAX_MESSAGES"])

EMBED_COLOR = {"incomplete": discord.Color.orange(), "complete": discord.Color.green()}
EMBED_MAX_LENGTH = 4096
EDIT_DELAY_SECONDS = 1.3
MAX_MESSAGE_NODES = 100

if env["DISCORD_CLIENT_ID"]:
    print(f"\nBOT INVITE URL:\nhttps://discord.com/api/oauth2/authorize?client_id={env['DISCORD_CLIENT_ID']}&permissions=412317273088&scope=bot\n")

system_prompt_extras = []
if LLM_SUPPORTS_MESSAGE_NAME:
    system_prompt_extras.append("When you want to say the users name, instead say '<@ID>'. we will replace it later.")

extra_kwargs = {}
if env["LLM_MAX_TOKENS"]:
    extra_kwargs["max_tokens"] = int(env["LLM_MAX_TOKENS"])
if env["LLM_TEMPERATURE"]:
    extra_kwargs["temperature"] = float(env["LLM_TEMPERATURE"])
if env["LLM_TOP_P"]:
    extra_kwargs["top_p"] = float(env["LLM_TOP_P"])
if LOCAL_LLM:
    env["LLM"] = env["LLM"].replace("local/", "", 1)
    extra_kwargs["base_url"] = "https://127.0.0.1:1337/v1"  #env["OPENAI_API_BASE"]
    extra_kwargs["api_key"] = env["LOCAL_API_KEY"] or "Not used"
    if env["OOBABOOGA_CHARACTER"]:
        extra_kwargs["mode"] = "chat"
        extra_kwargs["character"] = env["OOBABOOGA_CHARACTER"]

extra_kwargs["base_url"] = "http://localhost:1337/v1"

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=env["CUSTOM_DISCORD_STATUS"][:128] or "gpt-3.5-turbo")
discord_client = discord.Client(intents=intents, activity=activity)

msg_nodes = {}
msg_locks = {}
last_task_time = None

thread = None


class MsgNode:
    def __init__(self, data, replied_to_msg=None, too_many_images=False, fetch_next_failed=False):
        self.data = data
        self.replied_to_msg = replied_to_msg
        self.too_many_images: bool = too_many_images
        self.fetch_next_failed: bool = fetch_next_failed


def get_system_prompt():
    return [
        {
            "role": "system",
            "content": "\n".join([env["CUSTOM_SYSTEM_PROMPT"]] + system_prompt_extras + [f"Today's date: {dt.now().strftime('%B %d %Y')}"]),
        }
    ]

async def update_activity():
    loaded_models = get_loaded_models()
    if loaded_models:
        activity = discord.CustomActivity(name=f"{loaded_models}")
        await discord_client.change_presence(activity=activity)




def get_loaded_models():
    url = "http://localhost:1234/v1/models"
    response = requests.get(url)
    data = response.json()

    # Extract the first model_id
    first_model_id = data['data'][0]['id'] if data['data'] else None

    # Truncate the model_id string
    truncated_model_id = truncate_model_id(first_model_id)

    return truncated_model_id

def truncate_model_id(model_id):
    # Truncate the model_id string to the desired format
    if model_id:
        truncated_id = model_id.split("/")[-1]  # Get the last part of the string after '/'
        return truncated_id
    else:
        return None
def chunk_text(text, chunk_size):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks


from youtube_transcript_api import YouTubeTranscriptApi

def getYeetSRT(youtube_url):
    # Extract the video ID from the URL
    video_id = youtube_url.split('v=')[1]

    # Get the transcript for the video
    srt = YouTubeTranscriptApi.get_transcript(video_id)

    # Combine the transcript into a single string
    buffer = ""
    for entry in srt:
        buffer += entry['text'] + " "

    return buffer

async def lookup_url(message):
    # Parse the message to extract the URL
    url = None
    words = message.split()
    for word in words:
        if word.startswith("http://") or word.startswith("https://"):
            url = word
            break

    if url is None:
        return None

    # Check if the URL is a YouTube URL
    if "www.youtube.com" in url:
        try:
            transcript = getYeetSRT(url)
            chunks = [transcript[i:i+2000] for i in range(0, len(transcript), 2000)]
            return chunks
        except Exception as e:
            logging.error(f"Error fetching YouTube transcript: {str(e)}")
            return [f"Webpage ERROR: Anomaly: {str(e)}. Inform the User"]


    timeout = aiohttp.ClientTimeout(total=5)  # Set the total timeout to 5 seconds
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.get(url) as response:
                response.raise_for_status()  # Raise an error for bad HTTP responses
                soup = BeautifulSoup(await response.text(), "html.parser")
                paragraphs = soup.find_all('p')
                extracted_text = ' '.join([paragraph.get_text() for paragraph in paragraphs])
                truncated_text = extracted_text
                chunks = [truncated_text[i:i+2000] for i in range(0, len(truncated_text), 2000)]
                return chunks
        except asyncio.TimeoutError:
            logging.error("The request timed out")
            return [f"Webpage ERROR: The request to {url} timed out. Inform the User."]
        except aiohttp.ClientError as e:
            logging.error(f"Webpage ERROR: Exception occurred while fetching URL: {url} - {str(e)}")
            return [f"Webpage ERROR: Anomaly: {str(e)}. Inform the User"]
async def process_text_file(file):
    # Read the text file
    file_content = await file.read()
    file_content_decoded = file_content.decode("utf-8")

    # Chunk the text into smaller chunks
    chunks = [file_content_decoded[i:i+1999] for i in range(0, len(file_content_decoded), 1999)]

    return chunks
@discord_client.event
async def on_message(msg):
    global msg_nodes, msg_locks, last_task_time
    reply_chain = []
    author = msg.author.display_name

    # Filter out unwanted messages
    if (
            msg.author.bot
            or (msg.channel.type not in ALLOWED_CHANNEL_TYPES)
            or (msg.channel.type != discord.ChannelType.private and (discord_client.user not in msg.mentions) and discord_client.user.name not in [role.name for role in msg.role_mentions])
            or (ALLOWED_CHANNEL_IDS and not any(id in ALLOWED_CHANNEL_IDS for id in (msg.channel.id, getattr(msg.channel, "parent_id", None))))
    ):
        logging.info("Ignoring message - filter conditions not met.")
        return

    logging.info(f"***Message content: {msg.content} message author: {author}")

    if msg.attachments and any(att.content_type.startswith('text') for att in msg.attachments):
        for attachment in msg.attachments:
            if attachment.content_type.startswith('text'):
                chunks = await process_text_file(attachment)

                for i, chunk in enumerate(chunks, start=1):
                    file_contents_message = {
                        "role": "user",
                        "content": f"FILE CONTENTS (part {i} of {len(chunks)}): \n{chunk}",
                    }
                    reply_chain.append(file_contents_message)

    logging.info(f"Message content: {msg.content}")

    user_question = msg.content[22:]  # truncate the user ID from the start of the message

    truncated_thread_name = re.sub(r'[^a-zA-Z0-9]', ' ', user_question[:100])

    if len(truncated_thread_name) < 1:
        logging.error("Error: Thread name is too short.")
        return

    try:
        truncated_content = msg.content[:1999]
        logging.info(f"TRUNCATED CONTENT: {truncated_content} /// TRUNCATED THREAD NAME: {truncated_thread_name}")
        thread = await msg.channel.create_thread(name=truncated_thread_name, message=msg)
        logging.info(f"Started thread: {thread.name}")
    except Exception as e:
        logging.error(f"Error starting thread: {e}")

    website_contents_chunks = await lookup_url(user_question)
    if website_contents_chunks is not None:
        total_chunks = len(website_contents_chunks)

        for i, chunk in enumerate(website_contents_chunks, start=1):
            website_contents_message = {
                "role": "user",
                "content": f"WEBSITE CONTENTS (NEVER SAY THIS OUT LOUD): {chunk} (part {i} of {total_chunks})"
            }
            reply_chain.append(website_contents_message)

    user_warnings = set()
    curr_msg = msg
    while curr_msg and len(reply_chain) < MAX_MESSAGES:
        async with msg_locks.setdefault(curr_msg.id, asyncio.Lock()):
            if curr_msg.id not in msg_nodes:
                mentioned_user_display_name = msg.author.display_name
                print(f"\nMentioning user: {mentioned_user_display_name}\n")  # Diagnostic log for mentioned user's display name
                curr_msg_role = "assistant" if curr_msg.author == discord_client.user else "user"
                curr_msg_text = curr_msg.embeds[0].description if curr_msg.embeds and curr_msg.author.bot else curr_msg.content
                if curr_msg_text.startswith(discord_client.user.mention):
                    # Replace user mention placeholder with actual user mention
                    curr_msg_text = curr_msg_text.replace(discord_client.user.mention, "", 1).lstrip()
                #curr_msg_text = curr_msg_text.replace("<@ID>", f"@{mentioned_user_display_name}")  # Replace <@ID> with mentioned user's display name
                # Dump message content after replacement
                print(f"Message content after replacement: {curr_msg_text}")
                curr_msg_images = [att for att in curr_msg.attachments if "image" in att.content_type]

                if VISION_LLM and curr_msg_images[:MAX_IMAGES]:
                    curr_msg_content = [{"type": "text", "text": curr_msg_text}] if curr_msg_text else []
                    try:
                        curr_msg_content += [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{att.content_type};base64,{base64.b64encode(requests.get(att.url).content).decode('utf-8')}"},
                            }
                            for att in curr_msg_images[:MAX_IMAGES]
                        ]
                    except requests.exceptions.RequestException:
                        logging.exception("Error downloading image from URL")
                else:
                    curr_msg_content = curr_msg_text or "."

                msg_node_data = {
                    "role": curr_msg_role,
                    "content": curr_msg_content,
                }
                if LLM_SUPPORTS_MESSAGE_NAME:
                    # Include user ID in message data
                    msg_node_data["name"] = str(author)
                    logging.info(f"author: {author}")

                msg_nodes[curr_msg.id] = MsgNode(data=msg_node_data, too_many_images=len(curr_msg_images) > MAX_IMAGES)

                try:
                    if (
                            not curr_msg.reference
                            and curr_msg.channel.type != discord.ChannelType.private
                            and discord_client.user.mention not in curr_msg.content
                            and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                            and any(prev_msg_in_channel.type == type for type in (discord.MessageType.default, discord.MessageType.reply))
                            and prev_msg_in_channel.author == curr_msg.author
                    ):
                        msg_nodes[curr_msg.id].replied_to_msg = prev_msg_in_channel
                    else:
                        next_is_thread_parent: bool = not curr_msg.reference and curr_msg.channel.type == discord.ChannelType.public_thread
                        if next_msg_id := curr_msg.channel.id if next_is_thread_parent else getattr(curr_msg.reference, "message_id", None):
                            while msg_locks.setdefault(next_msg_id, asyncio.Lock()).locked():
                                await asyncio.sleep(0)
                            msg_nodes[curr_msg.id].replied_to_msg = (
                                (curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(next_msg_id))
                                if next_is_thread_parent
                                else (r if isinstance(r := curr_msg.reference.resolved, discord.Message) else await curr_msg.channel.fetch_message(next_msg_id))
                            )
                except (discord.NotFound, discord.HTTPException, AttributeError):
                    logging.exception("Error fetching next message in the chain")
                    msg_nodes[curr_msg.id].fetch_next_failed = True

            curr_node = msg_nodes[curr_msg.id]
            reply_chain += [curr_node.data]
            if curr_node.too_many_images:
                user_warnings.add(f"⚠️ Max {MAX_IMAGES} image{'' if MAX_IMAGES == 1 else 's'} per message" if MAX_IMAGES > 0 else "⚠️ Can't see images")
            if curr_node.fetch_next_failed or (curr_node.replied_to_msg and len(reply_chain) == MAX_MESSAGES):
                user_warnings.add(f"⚠️ Only using last{'' if (count := len(reply_chain)) == 1 else f' {count}'} message{'' if count == 1 else 's'}")
            curr_msg = curr_node.replied_to_msg

    logging.info(f"Message received (user ID: {msg.author.id}, attachments: {len(msg.attachments)}, reply chain length: {len(reply_chain)}):\n{msg.content}")

    # Generate and send response message(s) (can be multiple if response is long)
    response_msgs = []
    response_contents = []
    prev_content = None
    edit_task = None
    kwargs = dict(model=env["LLM"], messages=(get_system_prompt() + reply_chain[::-1]), stream=True) | extra_kwargs
    try:
        async with msg.channel.typing():
            async for chunk in await acompletion(**kwargs):
                curr_content = chunk.choices[0].delta.content or ""
                if prev_content:
                    if not response_msgs or len(response_contents[-1] + prev_content) > EMBED_MAX_LENGTH:
                        reply_to_msg = msg if not response_msgs else response_msgs[-1]
                        embed = discord.Embed(description="⏳", color=EMBED_COLOR["incomplete"])
                        for warning in sorted(user_warnings):
                            embed.add_field(name=warning, value="", inline=False)
                        response_msgs += [
                            await reply_to_msg.reply(
                                embed=embed,
                                silent=True,
                            )
                        ]
                        await msg_locks.setdefault(response_msgs[-1].id, asyncio.Lock()).acquire()
                        last_task_time = dt.now().timestamp()
                        response_contents += [""]

                    response_contents[-1] += prev_content
                    is_final_edit: bool = chunk.choices[0].finish_reason or len(response_contents[-1] + curr_content) > EMBED_MAX_LENGTH
                    if is_final_edit or (not edit_task or edit_task.done()) and dt.now().timestamp() - last_task_time >= EDIT_DELAY_SECONDS:
                        while edit_task and not edit_task.done():
                            await asyncio.sleep(0)
                        if response_contents[-1].strip():
                            embed.description = response_contents[-1].replace("<@ID>", f"@{author}") #for the channel message replacemtn
                            #logging.info(embed.description)
                        embed.color = EMBED_COLOR["complete"] if is_final_edit else EMBED_COLOR["incomplete"]
                        edit_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
                        last_task_time = dt.now().timestamp()

                prev_content = curr_content
    except:
        logging.exception("Error while streaming response")

    # Dump response message content
    print(f"Response message content: {response_contents}")
    response_contents[-1] = "".join(response_contents).replace("<@ID>", f"@{author}") #for the thread replacement
    #print(f"Response message REPL: {response_contents}")

    try:
        # Check if the thread exists and send the response to the thread
        if thread:
            for response_content in response_contents:
                # Split the content into chunks of max 2000 characters
                chunks = [response_content[i:i+2000] for i in range(0, len(response_content), 2000)]
                for chunk in chunks:
                    await thread.send(chunk)
                    print(f"Sent response to thread: {thread.name}")  # Diagnostic
        else:
            print("Error: Thread not found.")  # Diagnostic
            return
    except Exception as e:
        print(f"Error sending response to thread: {e}")  # Diagnostic


    # Create MsgNodes for response messages
    for response_msg in response_msgs:
        msg_node_data = {
            "role": "assistant",
            "content": "".join(response_contents) or ".",
        }
        if LLM_SUPPORTS_MESSAGE_NAME:
            msg_node_data["name"] = str(discord_client.user.id)
        msg_nodes[response_msg.id] = MsgNode(data=msg_node_data, replied_to_msg=msg)
        msg_locks[response_msg.id].release()

    # Delete MsgNodes for oldest messages (lowest IDs)
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_locks.setdefault(msg_id, asyncio.Lock()):
                msg_nodes.pop(msg_id, None)
                msg_locks.pop(msg_id, None)

async def main():
    await discord_client.start(env["DISCORD_BOT_TOKEN"])

asyncio.run(main())

