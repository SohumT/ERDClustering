from collections import Counter
import editdistance
from nltk.corpus import stopwords
import string
import sys

def calc_pr(A, B):
    intersection = list((Counter(A) & Counter(B)).elements())
    intersection_count = Counter(intersection)
    AB = 0
    for element, count in intersection_count.items():
        AB += count
        # print(f"Element: {element}, Count: {count}")

    precision = AB / len(A)
    recall = AB / len(B)
    f1score = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1score


def correct_misclassified(output_list, threshold, filtered_question_words):
    #  works better when threshold = 1. No change
    #  when threshold = 2, correct characters change to different words - ex: allows -> allowed, time->name, ..

    for l in range(len(output_list)):
        words = output_list[l][1:]

        '''if output_list[l][0] == "rel_attr" and len(words) > 1:
            output_list[l][1] = " ".join(words)
            output_list[l] = output_list[l][:2]'''

        for i in range(len(words)):
            if words[i] not in filtered_question_words and words[i].lower() not in filtered_question_words:  # this line prevents trip -> trips
                for j in range(len(filtered_question_words)):
                    if editdistance.eval(words[i], filtered_question_words[j]) <= threshold:
                        output_list[l][i + 1] = filtered_question_words[j]
                        break
    #print("After correcting: \n", output_list, "\n")
    return output_list


def main(output_list, ground_truth, question_path):
    #  This is tested with trip-1.txt and output.txt from discord

    with open(question_path, 'r', encoding="utf8") as file:
        question_doc = file.read()
    question_words = list(set(question_doc.split()))

    # remove stop words in the question text & remove punctuations of both ends (but not "-")
    stop_words = set(stopwords.words('english'))
    filtered_question_words = [question_words[i] for i in range(len(question_words)) if
                               question_words[i].lower() not in stop_words]
    filtered_question_words = list(set((word.strip(string.punctuation) for word in filtered_question_words)))
    # print(filtered_question_words)

    # correct some words returned by OCR
    threshold = 5
    #print("Before correcting: \n", output_list)
    final_output_list = correct_misclassified(output_list, threshold, filtered_question_words)

    if len(ground_truth) > 0:

        A = [word for sublist in final_output_list for word in sublist[1:]]
        B = [word for sublist in ground_truth for word in sublist[1:]]

        precision, recall, f1score = calc_pr(A, B)
        print(f"Threshold: {threshold}")
        print(f"Precision: {precision}\nRecall: {recall}\nF1 score: {f1score}")

        # precision: 0.84  Recall: 0.9130434782608695 when threshold = 1
        # precision: 0.8  Recall: 0.8695652173913043 when threshold = 2

    return final_output_list
    
