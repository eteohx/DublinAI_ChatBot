# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 12:40:22 2020

@author: Administrator
"""

from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from nlp_context_entity_wa import get_context_entity, last_context, last_entity
from movie_recommender_wa import movies_similar_to

app = Flask(__name__)

def reload_csv(user):
    session_df = pd.DataFrame({},columns=['context','entity']) #stores the session details of the user
    session_df.loc[0] = ['start_conversation','']
    session_df.to_csv('session_df' + str(user) + '.csv')

def reload_txt(user):
    f = open("convo" + str(user) + ".txt", "w")
    f.close() 
    
    
def chat_moviebot(message):
   output_str = ('<li class="in"><div class="chat-img"><img alt="Avtar" src="https://i.ibb.co/Q6pJTBZ/moviebot-icon.png"></div><div class="chat-body"><div class="chat-message"><h5 class="name">MovieBot</h5><p>' + message + '</p></div></div></li>')
#   output_str = ('<img src="https://i.ibb.co/Q6pJTBZ/moviebot-icon.png" alt="Avatar">' + new_message)
   return output_str

def chat_user(user,message):
  # new_message = '<div class="alert alert-secondary" role="alert"><b>' + user +'</b>' + message +" <br></div>"
#   output_str = ('<img src="https://i.ibb.co/Hq7rsfb/user-icon.png" alt="Avatar" class="right">' + new_message)
   output_str = ('<li class="out"><div class="chat-img"><img alt="Avtar" src="https://i.ibb.co/Hq7rsfb/user-icon.png"></div><div class="chat-body"><div class="chat-message"><h5>' + user + '</h5><p>' + message + '</p></div></div></li>')
   return output_str



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form['username']     
        reload_csv(user)
        reload_txt(user)
        return redirect(url_for('home', user=user))
    return render_template('login.html')


@app.route("/", methods=['GET','POST'])
def home():
    user = request.args.get('user')
 
    if request.method == 'POST':
        user_input = request.form['user_input']
        session_df = pd.read_csv('session_df'+user+'.csv',low_memory=False,index_col=0)
        store = 0
        context = ''
        entity = ''
        message = ''
        end_convo = 0
        context,entity = get_context_entity(user_input, session_df) # NLP          
        if context == 'information_genre':
            message = "Okay! Please name a movie you like."
            store = 1
        elif context == 'information_other_movie':
            genre = last_entity(session_df)
            # search database
            movie_list,urls,_ = movies_similar_to(entity,3,genre,method = 'collab',exclude_collection = True)
            movies = '<a href = "' + urls[-1] + '" class="text-danger"><b>' + movie_list[-1] + '</b></a>'
            for i in range(len(movie_list)-1):
                movies = '<a href = "' + urls[i] + '" class="text-danger"><b>' + movie_list[i] + '</b></a>, ' + movies
            message = "Got it! I think you might like these films - " + movies + ". Type another genre if you'd like more recommendations!"  
            end_convo = 1
        elif context == 'insult':
            if last_context(session_df) == 'information_genre':
                message = "That's not very nice, but I'll let it slide! :( Name a movie you like."   
            else:
                message = "That's not very nice! :( Let's start again. Name a genre." 
        elif context == 'incomprehensible':
            if last_context(session_df) == 'information_genre':
                message = "Sorry, I don't know that one! Name another movie please."
            else:
                message = "Sorry, I didn't understand that. Let's start again. Name a genre."

        f = open("convo"+str(user)+".txt", "a")
        #f.write('<div class="alert alert-secondary" role="alert"><b>' +user + "</b>: "+ user_input + " <br></div>")
        #f.write('<div class="alert alert-primary" role="alert"><b>MovieBot: </b>' + message + " <br></div>") 
        f.write(chat_user(user,user_input))
        f.write(chat_moviebot(message))
        f.close()
    
        if store:
            count = session_df.shape[0]
            session_df.loc[count] = [context,entity]
            session_df.to_csv('session_df'+ str(user) +'.csv')
        
        with open('convo' + str(user) +'.txt', 'r') as myfile:
            text = myfile.read()    
            
        if end_convo:
            reload_csv(user)
            end_convo = 0
            
        return render_template('chat_window.html',user=user,text_output=text)    # save context and entities to log
    return render_template("chat_window.html",user=user)


if __name__ == "__main__":
    app.run(port=5000)