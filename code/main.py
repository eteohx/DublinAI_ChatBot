# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:53:50 2020

@author: Emily

modified from: 
https://github.com/Sundar0989
https://towardsdatascience.com/chatbots-are-cool-a-framework-using-python-part-1-overview-7c69af7a7439

This code talks to the entire bot Framework. 
"""
from slackconfig import slack_client, slack_events_adapter
import pandas as pd

# Initialize with empty value to start the conversation.
user_input = ''
context = {}
current_action = ''
follow_ind = 0
session_df = pd.DataFrame({},columns=['timestamp', 'user', 'context']) #stores the session details of the user
bot_id = None

# constants
@slack_events_adapter.on("message")
def handle_message(event_data):
    message = event_data["event"]
    print(message.get('text'))
    print(message.get("subtype"))
    # If the incoming message contains "hi", then respond with a "Hello" message
    if message.get("subtype") is None and "hi" in message.get('text'):
        channel = message["channel"]
        message = "Hello <@%s>! :tada:" % message["user"]
        slack_client.chat_postMessage(channel=channel,text=message)

# Error events
@slack_events_adapter.on("error")
def error_handler(err):
    print("ERROR: " + str(err))
    
    
if __name__ == "__main__":
     print("MovieBot: Connected")
     slack_events_adapter.start(port=3000)    
     print("MovieBot: Disconnected")
