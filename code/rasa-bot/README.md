You can run the chatbot including the sentiment analysis by running the following three commands in three different terminal windows while being in the child_avatar folder: 
- rasa run -m models --endpoints endpoints.yml --port 5002 --credentials credentials.yml
- rasa run actions
- python3 start.py

By running these, you will be able to get both the sentiment of your input and the sentiment of the bot'
s response. 

It is also possible to run "rasa shell" in your terminal. This way you will be able to get in a conversation with the chatbot without seeing the sentiment results. 

Lastly, you can run "rasa shell nlu". Running this command will get you in the NLU interface of RASA. Using this interface you will be able to see only the sentiment entity of your input (including all other model intents).
