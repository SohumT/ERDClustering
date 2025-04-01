from ultralytics import YOLO
import easyocr
import numpy as np
import cv2
import csv
import argparse
import edit_distance
import nltk
import os

def get_coordinates(model, image_path):
    results = model.predict(image_path)

    coordinates = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            names = model.names
            x1, y1, x2, y2 = box.xyxy[0]
            class_id = names[int(box.cls)]
            coordinates.append([class_id, x1, y1, x2, y2])

    return coordinates

def ocr(image_path, data):

    reader = easyocr.Reader(['en'])

    img = cv2.imread(image_path)
    result_data = []

    for item in data:
        class_name, x1, y1, x2, y2 = item

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        roi = img[y1:y2, x1:x2]

        # Grayscale, Gaussian blur, Otsu's threshold
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # blur = cv2.GaussianBlur(gray, (3,3), 0)
        # thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        ret, thresh = cv2.threshold(gray, 13, 255, cv2.THRESH_BINARY)

        # Morph open to remove noise and invert image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        invert = 255 - opening

        result = reader.readtext(invert)

        # Extract and format
        if result:
            extracted_texts = [class_name]
            combined_text = ''
            prev_text = ''

            for _, text, _ in result:
                
                '''if prev_text.endswith('_'):
                    combined_text = prev_text + text
                elif class_name == 'rel_attribute':
                    if combined_text:
                        combined_text += ' ' + text
                        combined_text = combined_text.replace('- ', '-').replace(' -', '-')
                    else:
                        combined_text = text
                else:'''
                if combined_text: # indent later here to repair script
                    extracted_texts.append(combined_text)
                combined_text = text

                prev_text = text

            if combined_text:
                extracted_texts.append(combined_text)
            result_data.append(extracted_texts)
    
    return result_data



def main(model, image_path, ground_truth, question_path):
    
    coordinates = get_coordinates(model, image_path)
    
    # print(coordinates)
    ocr_returns = ocr(image_path, coordinates)
    # print(ocr)

    # Used if we have the ground truth
    return edit_distance.main(ocr_returns, ground_truth, question_path)

def getOCRAdjusted(model, image_path, question_path):

    print(image_path)
    coordinates = get_coordinates(model, image_path)
    
    # print(coordinates)
    #print("regular")
    #print(image_path, coordinates)
    ocr_returns = ocr(image_path, coordinates)
    #print("ocr returns")
    #print(ocr_returns)
    # print(ocr)

    # used if we do not have ground truth
    output_list = edit_distance.main(ocr_returns, [], question_path)

    # insert post processing and return output list 

    underscore = False

    for i in range(len(output_list)):
        for j in range(len(output_list[i])):

            output_list[i][j].find('_')
            underscore = True

    for i in range(len(output_list)):
        
        j = 1

        while j < len(output_list[i]) - 1:

            if (output_list[i][0] == 'rel_attr'):
                output_list[i][j] = output_list[i][j] + ' ' + output_list[i][j+1]
                output_list[i][j] = output_list[i][j].replace('- ', '-').replace(' -', '-')
                output_list[i].pop(j+1)
            elif (output_list[i][0] == 'rel' or output_list[i][0] == 'ident_rel'):
                if underscore:
                    output_list[i][j] = output_list[i][j] + '_' + output_list[i][j+1]
                else:
                    output_list[i][j] = output_list[i][j] + ' ' + output_list[i][j+1]
                output_list[i].pop(j+1)
                # Unsure of whether to include this change
            elif output_list[i][j].endswith('_') or output_list[i][j].endswith('__'):
                
                output_list[i][j] = output_list[i][j] + output_list[i][j+1]
                output_list[i].pop(j+1)

            elif output_list[i][j].endswith('-'):
            
                output_list[i][j] = output_list[i][j] + output_list[i][j+1]
                output_list[i].pop(j+1)
                
            j+=1
    
    #print("after edit distance")'''
    return output_list

    #return edit_distance.main(ocr_returns, [], question_path)

def getOCRAdjustedRoboflow(coordinates, image_path, question_path):
    
    # print(coordinates)
    ocr_returns = ocr(image_path, coordinates)
    #print("ocr returns")
    #print(ocr_returns)
    # print(ocr)

    # used if we do not have ground truth
    output_list = edit_distance.main(ocr_returns, [], question_path)

    # insert post processing and return output list 

    for i in range(len(output_list)):
        
        j = 1

        while j < len(output_list[i]) - 1:

            if (output_list[i][0] == 'rel_attr'):

                output_list[i][j] = output_list[i][j] + ' ' + output_list[i][j+1]
                output_list[i][j] = output_list[i][j].replace('- ', '-').replace(' -', '-')
        
                output_list[i].pop(j+1)
            
            # Unsure of whether to include this change
            elif output_list[i][j].endswith('_'):
                
                output_list[i][j] = output_list[i][j] + output_list[i][j+1]
                output_list[i].pop(j+1)

            elif output_list[i][j].endswith('-'):
            
                output_list[i][j] = output_list[i][j] + output_list[i][j+1]
                output_list[i].pop(j+1)
                
            j+=1
    
    #print("after edit distance")
    return output_list


if __name__ == "__main__":

    model = YOLO('./yolov8m.pt')
    question_path = './OD_OCR_testing/video_games/Question.txt'
    output_path = './OCR_Outputs'
    image_path = './OD_OCR_testing/video_games/8.png'    
    img_name = '8'

    ground_truth = [['entity', 'developer', 'PK', 'id', 'name', 'headquarter'],
    ['entity', 'region', 'PK', 'id', 'name', 'population'],
    ['entity', 'video_game', 'PK', 'rank', 'name', 'year', 'genre'],
    ['entity', 'platform', 'PK', 'platform_id', 'platform_name', 'introductory_price_us', 'units_sold'],
    ['weak_entity', 'games_sales', 'sales'],
    ['rel', 'publish'],
    ['rel', 'develop'],
    ['rel', 'Run on'],
    ['ident_rel', 'Sell']]
        
    final_output_list = main(model, image_path, ground_truth, question_path)
    print(final_output_list)
    save_path = './OD_OCR_Output'

    completeName = os.path.join(save_path, img_name+".txt") 
    with open(completeName, "w") as file:
        writer = csv.writer(file)
        for item in final_output_list:
            # print(item)
            formatted_item = ', '.join(['\'' + str(i) + '\'' for i in item])
            file.write('[' + formatted_item + ']\n')

    print(f"OCR output saved to {file}")

    # document_vectors = []
    # for item in final_output_list:
    #     document_vector = item[1:]
    #     document_vectors.append(document_vector)

    # for i, document_vector in enumerate(document_vectors, start=1):
    #     print(f"Document {i}: {document_vector}")