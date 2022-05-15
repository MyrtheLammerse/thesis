from transformers import AutoTokenizer, AutoModel
from torch.nn import functional as F
import csv

tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
model = AutoModel.from_pretrained('facebook/bart-large-mnli')

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

def words_removal(sentence):
    stopwords = ["okay", "hm-hm", "thank you"]
    querywords = sentence.split()
    resultwords = [word for word in querywords if word.lower() not in stopwords]
    return ' '.join(resultwords)

if __name__ == "__main__":
    window_size = 7
    overall_count = 0
    part_story_list = []
    prev_transcript_id = None
    remove_stopwords = False
    # we open the csv file that has the transcript(s)
    with open('/Users/Myrthe/Downloads/Thesis/SentimentAnalysis/complete_survey_excerpts_window7-1.csv') as csv_input:
        # create an output file for said transcript(s)
        with open('results-cosine/facebook/bart_large_mnli/complete_survey_excerpts_window7-1_interviewer_facebook-bart-large-mnli.csv', 'w') as csv_output:
            writer = csv.writer(csv_output)
            reader = csv.reader(csv_input, delimiter=';')

            all_rows = []
            row = next(reader)
            # the following headers are added to the csv file
            row.extend(["Sentiment (window size 7 / no threshold)"])

            all_rows.append(row)
            # for each row in the csv file, we get the transcript_id first. Then we classify the child's response.
            for row in reader:
                transcript_id = row[0]
                overall_count += 1
                all_rows.append(row)
                row1_lower = row[1].lower()
                row2_lower = row[2]
                # predict the sentiment based on the sentences within the window size.
                # we are appending the response to a list, but if the length of the list is bigger than the window size,
                # we will pop the first item before adding a new one.
                if remove_stopwords:
                    row1_lower = words_removal(row1_lower)
                    row2_lower = words_removal(row2_lower)
                if transcript_id == prev_transcript_id:
                    if len(part_story_list) < window_size:
                        part_story_list.append(row1_lower+"? "+row2_lower+". ")
                    else:
                        part_story_list.pop(0)
                        part_story_list.append(row1_lower+"? "+row2_lower+". ")
                else:
                    part_story_list.clear()
                    part_story_list.append(row1_lower+"? "+row2_lower+". ")

                # join all the responses together to be classified as a whole.
                part_story = "".join(part_story_list)
                sentiment_part_story_pred = classifier(part_story)
                row.append(sentiment_part_story_pred)
                if overall_count % 50 == 0:
                    print(overall_count)
                prev_transcript_id = transcript_id

            writer.writerows(all_rows)
            print("done")

