from transformers import AutoTokenizer, AutoModel
from torch.nn import functional as F
import csv

tokenizer = AutoTokenizer.from_pretrained('deepset/sentence_bert')
model = AutoModel.from_pretrained('deepset/sentence_bert')

labels = ['fear', 'enjoyment', 'sadness', 'anger']


def classifier(sentence):
    # run inputs through model and mean-pool over the sequence
    # dimension to get sequence-level representations
    inputs = tokenizer.batch_encode_plus([sentence] + labels,
                                         return_tensors='pt',
                                         padding=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    output = model(input_ids, attention_mask=attention_mask)[0]
    sentence_rep = output[:1].mean(dim=1)
    label_reps = output[1:].mean(dim=1)

    # now find the labels with the highest cosine similarities to the sentence
    similarities = F.cosine_similarity(sentence_rep, label_reps)

    # closest = similarities.argsort(descending=True)
    # for ind in closest:
    #     print(f'label: {labels[ind]} \t similarity: {similarities[ind]}')
    return labels, similarities


singlesentences = ["Then he tried to push it in and it really hurt.",
"And then I got taken to the principal.",
"He starts shouting and he was swearing and then he hit Mum.",
"He hurt mum and me.",
"We watched a movie and then we got some ice cream and then we went to bed.",
"He told us that we had to go into the cubicle one at a time.",
"Because I was hiding there.",
"It really hurt."]


if __name__ == "__main__":
    window_size = 3
    overall_count = 0
    part_story_list = []
    prev_transcript_id = None
    s = True
    if s:
        for sentence in singlesentences:
            print(sentence)
            print(classifier(sentence))
        quit()

    # we open the csv file that has the transcript(s)
    with open('/Users/Myrthe/Downloads/Thesis/SentimentAnalysis/complete_survey_excerpts_window3.csv') as csv_input:
        # create an output file for said transcript(s)
        with open('/Users/Myrthe/Documents/thesis-pytorch/test.csv', 'w') as csv_output:
            writer = csv.writer(csv_output)
            reader = csv.reader(csv_input, delimiter=',')

            all_rows = []
            row = next(reader)
            # the following headers are added to the csv file
            row.extend(["Sentiment (window size 3 / no threshold)"])

            all_rows.append(row)

            # for each row in the csv file, we get the transcript_id first. Then we classify the child's response.
            for row in reader:
                transcript_id = row[0]
                overall_count += 1
                all_rows.append(row)
                # predict the sentiment based on the sentences within the window size.
                # we are appending the response to a list, but if the length of the list is bigger than the window size,
                # we will pop the first item before adding a new one.
                if transcript_id == prev_transcript_id:
                    if len(part_story_list) < window_size:
                        part_story_list.append(row[2])
                    else:
                        part_story_list.pop(0)
                        part_story_list.append(row[2])
                else:
                    part_story_list.clear()
                    part_story_list.append(row[2])

                # join all the responses together to be classified as a whole.
                part_story = ". ".join(part_story_list)
                sentiment_part_story_pred = classifier(part_story)
                row.append(sentiment_part_story_pred)
                if overall_count % 50 == 0:
                    print(overall_count)
                prev_transcript_id = transcript_id

            writer.writerows(all_rows)
            print("done")

