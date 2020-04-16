# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:54:21 2020

@author: Emily Teoh
"""

import os
import slack
import nest_asyncio
from slackeventsapi import SlackEventAdapter
import nest_asyncio

def parse_bot_commands(event_data):
    message = event_data["event"]
    text = message.get('text')
    channel = message["channel"]
    timestamp = message['ts']
    user = message["user"]
    return text, channel, timestamp, user, message

# Our app's Slack Event Adapter for receiving actions via the Events API
location = "/G:/My Drive/DublinAI/Mini Projects/chatbot/"  # replace with the full folder path where you downloaded the github repo
nest_asyncio.apply()

slack_bot_token = os.environ['SLACK_BOT_TOKEN']
slack_verification_token = os.environ['SLACK_VERIFICATION_TOKEN']
slack_signing_secret = os.environ["SLACK_SIGNING_SECRET"]

slack_client = slack.WebClient(token=slack_bot_token)
slack_events_adapter = SlackEventAdapter(slack_signing_secret, "/slack/events")





