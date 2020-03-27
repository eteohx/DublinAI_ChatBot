# ChatBot - Recommender



## Background

A chatbot refers to software that can &#39;converse&#39; via written text or auditory methods for the purposes of retrieving information and simulating an interaction with a real person.

Chatbots fall into several different classes:

<img src="https://github.com/eteohx/DublinAI_ChatBot/blob/master/images/conversation_framework.png" width="700" height="400">

Open domain bots are not topic specific (e.g. Siri, Google Assistant) and try to imitate humanlike conversation. But they cannot answer a specific, domain-based question.

Closed domain bots answer specific questions. Rule-based ones are the simplest to build in that they retrieve pre-set responses. Generative-based ones try to imitate an agent while answering the customer&#39;s question.

There are several components to building a chat bot – the user interface, the NLP layer, the knowledge base and the data store

<img src="https://github.com/eteohx/DublinAI_ChatBot/blob/master/images/components.png" width="650" height="400">

For a closed-domain recommender bot (as we&#39;re considering here), it needs to understand the following to respond to a user question:

1. (Intent) What is the user talking about
2. (Entities) Did the user mention anything specific?
3. (Dialogue) What other information do we need? What should the bot ask to get these details?
4. (Request Fulfillment) How to fulfil the user request?

The NLP layer needs to be able to extract and generate that information.



## Problem Statement

Customers have a lot of choice when it comes to selecting a movie to watch. Can we build a ChatBot to offer them options based on their stated preferences?



## Scope

We will start with a closed domain, rule-based bot for movie recommendation using a Kaggle dataset. There are some useful tutorials doing this, and APIs for the UI and NLP layers. These tools will be used as a starting point. We will then try some custom NLP.

## Resources

(Tutorials)

[https://towardsdatascience.com/chatbots-are-cool-a-framework-using-python-part-1-overview-7c69af7a7439](https://towardsdatascience.com/chatbots-are-cool-a-framework-using-python-part-1-overview-7c69af7a7439)

[https://www.datacamp.com/community/tutorials/recommender-systems-python](https://www.datacamp.com/community/tutorials/recommender-systems-python)

[https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077](https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077)

[https://www.ibm.com/watson/how-to-build-a-chatbot](https://www.ibm.com/watson/how-to-build-a-chatbot)

(Front end (UI))

Slack (looking into other options)

(NLP)

IBM Watson, Python

(Data)

[https://www.kaggle.com/rounakbanik/movie-recommender-systems/data](https://www.kaggle.com/rounakbanik/movie-recommender-systems/data)

## Deliverables

Bot (probably on Slack) that will interact with a user to figure out which movie(s) would best suit their preferences

## Extensions

Maybe to books or music. Datasets:[https://github.com/caserec/Datasets-for-Recommender-Systems](https://github.com/caserec/Datasets-for-Recommender-Systems)

