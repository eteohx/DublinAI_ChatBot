# DublinAI_ChatBot

A project exploring building a chatbot for making movie recommendations to users


Current progress: 
- Front end: Designed a simple web chat UI; also able to deploy the chatbot on Slack on my personal workspace
- NLP: The bot asks user two questions - their preferred genre of movie, and to name another movie that they like. In the database, we have 20 different genres and ~4000 movie titles. To account for the fact that the user might use different terminology than the entities in the database (e.g. they might say 'scary movie' instead of 'horror'), we check the cosine similarity between the GloVe vectors of the user's input and all entities.
- Collaborative filtering recommender system: An embedded neural network was trained on a large set of user ratings to predict individual user movie ratings. RMSE of this model is ~0.81 (comparable with state-of-the-art). The trained movie embeddings appear to capture similarity - e.g. Pixar movies are close together in the vector space, movies from the same collection are also close together. We evaluated how well we can predict a user's rating of a particular movie by weighting the user's ratings of other movies by the cosine distance of the embedding vectors of those movies to the movie in question. This resulted in an RMSE of ~0.94. This is (as one would expected) poorer than our neural network, but still suggests that the embeddings capture something about users' tastes in movies. 
We used these trained embeddings in our chatbot.
- Evaluation for (simulated) cold-start users: Using just the embeddings, we find k closest movies to a particular movie that has been watched/rated by a user. We then check how many of these k movies have actually been watched by the user.

Slack-interface:

<img src="https://github.com/eteohx/DublinAI_ChatBot/blob/master/reports/images/test_bot.PNG" width="500" height="350">
