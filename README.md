# DublinAI_ChatBot

A project exploring building a chatbot for making recommendations to customers


Current progress: 
- Trying out collaborative filtering recommender algorithms to get embeddings/factorizations for movies - RMSE is ~0.81 for the neural network
- Plan is to then use the movie embeddings to pick the closest movie to a movie that our cold-start user likes


At the moment, basic bot works using content-based filtering:
<img src="https://github.com/eteohx/DublinAI_ChatBot/blob/master/reports/images/test_bot.PNG" width="500" height="350">


Data:
https://www.kaggle.com/rounakbanik/the-movies-dataset
