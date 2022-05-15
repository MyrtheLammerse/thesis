import requests
from actions.sentiment import SentimentAnalyzer

message = ""
bot_message = ""

analyser = SentimentAnalyzer()
print("Welcome to the child avatar, you can start the conversation. You can stop the chatbot by entering \"\stop\".")
while message != "\stop":
    message = input()
    if len(message) == 0:
        continue

    r = requests.post("http://localhost:5002/webhooks/rest/webhook", json={"message": message})
    print("message sentiment: ", analyser.sentiment(message))
    for i in r.json():
        bot_message = i['text']

        print(f"{bot_message}")
        print("bot response sentiment: ", analyser.sentiment(bot_message))
