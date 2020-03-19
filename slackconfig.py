# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:54:21 2020

@author: Emily Teoh
"""

import slack
import os
import nest_asyncio

location = "/G:/My Drive/DublinAI/Mini Projects/chatbot/"  # replace with the full folder path where you downloaded the github repo
nest_asyncio.apply()

###################################################################
######## Slack configuration   ##########################
###################################################################
slack_bot_token = os.environ['SLACK_BOT_TOKEN']
slack_verification_token = os.environ['SLACK_VERIFICATION_TOKEN']

#client = slack.WebClient(token=SLACK_BOT_TOKEN)
slack_client = slack.WebClient(token=slack_bot_token)

#client.chat_postMessage(
#  channel="#general",
#  text="testing :tada:"
#)
#
