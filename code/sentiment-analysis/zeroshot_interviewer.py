from transformers import pipeline
import csv

# create the zero shot classifier
classifier = pipeline("zero-shot-classification", model="Narsil/deberta-large-mnli-zero-cls")

# candidate class labels
class_labels = ["negative", "neutral", "positive"]
sentiment_labels = ['enjoyment', 'fear', 'sadness', 'anger']

def words_removal(sentence):
    stopwords = ["okay", "hm-hm", "thank you"]
    querywords = sentence.split()
    resultwords = [word for word in querywords if word.lower() not in stopwords]
    return ' '.join(resultwords)

if __name__ == "__main__":
    window_size = 3
    overall_count = 0
    part_story_list = []
    prev_transcript_id = None
    remove_stopwords = False
    # we open the csv file that has the transcript(s)
    with open('/Users/Myrthe/Downloads/Thesis/SentimentAnalysis/complete_survey_excerpts_window3.csv', encoding='utf-8') as csv_input:
        # create an output file for said transcript(s)
        with open('results-zeroshot/Narsil/deberta-large-mnli-zero-cls/complete_survey_excerpts_window3_interviewer_narsil-deberta_large_mnli_zero_cls.csv', 'w') as csv_output:
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
                sentiment_part_story_pred = classifier(part_story, sentiment_labels)
                row.append(sentiment_part_story_pred)
                if overall_count % 50 == 0:
                    print(overall_count)
                prev_transcript_id = transcript_id

            writer.writerows(all_rows)
            print("done")

