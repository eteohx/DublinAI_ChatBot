# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:53:50 2020

@author: Emily

partly based on:
https://github.com/Sundar0989
https://towardsdatascience.com/chatbots-are-cool-a-framework-using-python-part-1-overview-7c69af7a7439

This code talks to the entire bot Framework. 
"""
from slackconfig import slack_client, slack_events_adapter, parse_bot_commands
import pandas as pd
from nlp_context_entity import get_context_entity, last_context, last_entity
from movie_recommender import movies_similar_to

# Initialize with empty value to start the conversation.
session_df = pd.DataFrame({},columns=['timestamp', 'user', 'context','entity']) #stores the session details of the user
session_open = 0

# constants
@slack_events_adapter.on("message")
def handle_message(event_data):    
    global bot_id
    global session_open
    global session_df
    store = 0
    context = ''
    entity = ''
    # parse message
    text,channel,timestamp,user,full_message = parse_bot_commands(event_data)

    if not session_open:
       if bot_id in text:
                message = "Hello <@%s>! :tada: I'm MovieBot and I can recommend a movie to you. What genre are you in the mood for?" % user
                slack_client.chat_postMessage(channel=channel,text=message)
                context = 'start_conversation'
                session_open = 1
                store = 1
    else:  
        context,entity = get_context_entity(text, session_df, user) # NLP          
        if context == 'information_genre':
           message = "Okay! Please name a movie you like."
           slack_client.chat_postMessage(channel=channel,text=message)
           store = 1
        elif context == 'information_other_movie':
            genre = last_entity(session_df,user)
            # search database
            movies = movies_similar_to(genre,entity)
            message = "Got it! I think you might like these films: %s. Have a nice day!" % movies.title()    
            slack_client.chat_postMessage(channel=channel,text=message)
            session_open = 0 
            store = 1
        elif context == 'insult':
            if last_context(session_df,user) == 'information_genre':
                message = "<@%s>, that's not very nice, but I'll let it slide! :( Name a movie you like."   % user
            else:
                message = "<@%s>, that's not very nice! :( Let's start again. Name a genre."
            slack_client.chat_postMessage(channel=channel,text=message)        
        elif context == 'incomprehensible':
            if last_context(session_df,user) == 'information_genre':
                message = "Sorry, I don't know that one! Name another movie please."   % user
            else:
                message = "Sorry, I didn't understand that. Let's start again. Name a genre."
            slack_client.chat_postMessage(channel=channel,text=message)
                
    # save context and entities to log
    
    if store:
        count = session_df.size
        session_df.loc[count] = [timestamp,user,context,entity]
        
# Error events 
@slack_events_adapter.on("error")
def error_handler(err):
    print("ERROR: " + str(err))
    
if __name__ == "__main__":
     print("MovieBot: Connected")
     bot_id = slack_client.api_call("auth.test")["user_id"]
     print('Bot ID: ' + bot_id)
     slack_events_adapter.start(port=3000)    
     print("MovieBot: Disconnected")
