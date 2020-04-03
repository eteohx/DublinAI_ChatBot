# DublinAI_ChatBot

A project exploring building a chatbot for making recommendations to customers


Current progress: 
- Trying out collaborative filtering recommender algorithms to get embeddings/factorization for movies - RMSE is ~0.81 for the neural network
- Plan is to then use the movie embeddings to pick the closest movie to a movie that our cold-start user likes

<img src="https://github.com/eteohx/DublinAI_ChatBot/blob/master/reports/images/test_bot.PNG" width="500" height="350">


Data:
https://www.kaggle.com/rounakbanik/the-movies-dataset

Libraries: 
slackapi, SpaCy
