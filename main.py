# Final Model Parameters for OCR

from ultralytics import YOLO
import ocr
import os
import glob
import csv
import numpy as np
from PIL import Image

def trainModel(model_name, epochs, dataset_yaml_path):

    model = YOLO(model_name)
    #batch = ultralytics.utils.autobatch.autobatch(model, imgsz=640, fraction=0.6, batch_size=16)
    trained_model = model.train(data=dataset_yaml_path, batch=8, epochs=epochs, project='OD_Model') 
    return trained_model

def predictImageOD(model, img_path):

    # Model must be in Yolo(modle_path) form
    return model(img_path, save=True)


def getOutput(dir_path, output_path, model):
    # Specify the directory path you want to search in
    directory_path = dir_path

    # Use os.listdir() to get a list of all items (files and directories) in the specified directory
    all_items = os.listdir(directory_path)

    # Use a list comprehension to filter out only the directories
    directories = [item for item in all_items if os.path.isdir(os.path.join(directory_path, item))]

    for dir in directories:
        question_path = directory_path + "/Question.txt"

        png_files = glob.glob(os.path.join(os.path.join(directory_path, dir), '*.png'))

        for png in png_files:

            png_name = os.path.basename(png).split(".")[0]
            # question_path = os.path.join(directory_path, dir) + "/question.txt" # for stage1

            final_output_list = ocr.getOCRAdjusted(model, png, question_path)

            save_path = output_path

            completeName = os.path.join(save_path, png_name+".txt")

            with open(completeName, "w") as file:
                writer = csv.writer(file)
                for item in final_output_list:
                    # print(item)
                    formatted_item = ', '.join(['\'' + str(i) + '\'' for i in item])
                    file.write('[' + formatted_item + ']\n')

            print(f"OCR output saved to {file}")

def getOutput2(dir_path, output_path, model):
    # Specify the directory path you want to search in
    directory_path = dir_path

    # Use os.listdir() to get a list of all items (files and directories) in the specified directory
    all_items = os.listdir(directory_path)

    # Use a list comprehension to filter out only the directories
    directories = [item for item in all_items if os.path.isdir(os.path.join(directory_path, item))]

    for dir in directories:
        question_path = directory_path + "/Question.txt"

        png_files = glob.glob(os.path.join(directory_path, '*.png'))

        for png in png_files:

            png_name = os.path.basename(png).split(".")[0]
            # question_path = os.path.join(directory_path, dir) + "/question.txt" # for stage1

            final_output_list = ocr.getOCRAdjusted(model, png, question_path)

            save_path = output_path

            completeName = os.path.join(save_path, png_name+".txt")

            with open(completeName, "w") as file:
                writer = csv.writer(file)
                for item in final_output_list:
                    # print(item)
                    formatted_item = ', '.join(['\'' + str(i) + '\'' for i in item])
                    file.write('[' + formatted_item + ']\n')

            print(f"OCR output saved to {file}")



    
def add_padding(image_dir_path, padding_size=20):
    for img in os.Path(image_dir_path).iterdir():
        image = Image.open(img)
        width, height = image.size
        new_width = width + 2 * padding_size
        new_height = height + 2 * padding_size
        new_image = Image.new("RGB", (new_width, new_height), "white")
        new_image.paste(image, (padding_size, padding_size))
        new_image.save(img)


    
if __name__ == "__main__":

    # If model is untrained use this line of code 
    #model = trainModel('yolov8x.pt', 300, './dataset/data.yaml')

    # If model is trained use this line of code:
    model = YOLO('kyobest.pt')
    
    #Output from current model
    getOutput('./dataset1_K_4', './OD_OCR_Output', model)
