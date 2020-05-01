# MovieBot

This project explores building a chatbot application that makes movie recommendations to users.

Video explanation: https://www.loom.com/share/941c76accfea4fb7a7660d733f918235

Web application demo: http://34.244.10.159/login (deployed on an AWS virtual machine)


## Folders and files in this repo

* [Web application](https://github.com/eteohx/DublinAI_ChatBot/tree/master/code/web_application):
  * app.py: Flask application with routes to a login page and a chat page, where it executes a simple dialogue flow through state management 
  * nlp_context_entity_wa.py: NLP to link users' answers to entities in the database
  * movie_recommender_wa.py: Recommender system that takes in entities as input and returns k movies that are most suitable based on  cosine similarity of movie embedding vectors. The embedding vectors were produced by training an embedded neural network to predict user ratings.
* [Slack application](https://github.com/eteohx/DublinAI_ChatBot/tree/master/code/slack_application): As above, but with code for interfacing with Slack (receiving events via the Slack Events API and sending messages via Slack Web API). To set up a Slack Bot, see: https://github.com/slackapi/python-slack-events-api
* [Embedded Neural Network](https://github.com/eteohx/DublinAI_ChatBot/tree/master/code/recommender_embedded_nn): Code for setting up and training an embedded neural network to predict user ratings (collaborative filtering). RMSE of this model is ~0.81 (comparable with state-of-the-art). 
* [Evaluation of chatbot recommender system](https://github.com/eteohx/DublinAI_ChatBot/blob/master/code/movie_recommender_evaluation.ipynb): We test the system on unseen user rating data. We input one movie watched and rated positively by each user, and evaluate the precision of the three recommended outputs of the system. 
* [Data munging](https://github.com/eteohx/DublinAI_ChatBot/blob/master/code/preprocessData.ipynb): Pre-processing of movie metadata
* [Visualising Movie Embeddings](https://github.com/eteohx/DublinAI_ChatBot/blob/master/code/visualise_embeddings.ipynb): Projecting the movie embedding vectors to two dimensions and visualising how movies with various features are distributed in this vector space

## Slack interface

<img src="https://github.com/eteohx/DublinAI_ChatBot/blob/master/reports/images/test_bot.PNG" width="500" height="350">

## Resources 
* [Movielens dataset](https://grouplens.org/datasets/movielens/)
* [Implementing a recommender system using deep learning](https://medium.com/@iliazaitsev/how-to-implement-a-recommendation-system-with-deep-learning-and-pytorch-2d40476590f9)
* [Setting up a Slack Bot](https://github.com/slackapi/python-slack-events-api)
* [Bootstrap CSS Library](https://getbootstrap.com/)

<sup>Cleaned datasets were too large to be uploaded, but can be provided upon request</sup>

