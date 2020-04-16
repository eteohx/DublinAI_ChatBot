# DublinAI_ChatBot

A project exploring building a chatbot for making recommendations to customers


Current progress: 
- Obtained movie embeddings from a neural network trained to predict user movie ratings (trained only on a large set of ratings). The embedding vectors appear to capture similarity - e.g. movies of a similar genre appear to be close together in the vector space
- How good of a recommendation can we give simply using the distance of embedding vectors? Currently evaluating this.. So far have tried predicting user ratings by weighting them using distance in embedding space (RMSE), as well as computing precision at k

With the embedding vectors, we can now do both collaborative and content-based filtering.

<img src="https://github.com/eteohx/DublinAI_ChatBot/blob/master/reports/images/test_bot.PNG" width="500" height="350">
