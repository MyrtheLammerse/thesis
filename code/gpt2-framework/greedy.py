import torch

def greedy_search(model, tokenizer):
    print("Start the conversation. Write ! to quit.")
    step = 0
    text = ""
    while text != "!":
        if text != "!":
            # take user input
            text = input(">> You: ")
            if text == "!":
                break
            # encode the input and add end of string token
            input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")
            # concatenate new user input with chat history (if there is)
            bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if step > 0 else input_ids
            # generate a bot response
            if bot_input_ids.shape[1] > 1000:
                bot_input_ids = input_ids
            chat_history_ids = model.generate(
                bot_input_ids,
                max_length=1000,
                pad_token_id=tokenizer.eos_token_id
            )
            #print the output
            output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
            print(f"Bot: {output}")
            step = 1