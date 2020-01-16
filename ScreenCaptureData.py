import io
import io
import smtplib
import time
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import cv2
import keyboard
import numpy as np
import pandas as pd
import pyscreenshot as ImageGrab
import pytesseract


def diff_images(image_a, image_b):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((image_a.astype("float") - image_b.astype("float")) ** 2)
    err /= float(image_a.shape[0] * image_a.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def is_not_bracket(label):
    return (label != "[" and label != "]")

def is_bracket(label):
    return (label == "[" or label == "]")

def get_label_string(y):
    if type(y) is str:
        y = y.replace("[", " [ ").replace("  ", " ")
        y = y.replace("]", " ] ").replace("  ", " ")
        y = y.strip().split(" ")
        y = [y]

    labels = []

    for y_elem in y:
        text = ""
        i = 0
        prev_label = ""

        for y_label in y_elem:
            label = y_label.strip(" ")
            #print(prev_label, label)
            if is_bracket(label) and is_bracket(prev_label):
                spnsp = ""
            elif (is_not_bracket(label) and prev_label=="]") or (is_not_bracket(label) and is_not_bracket(prev_label)) or (label == "[" and is_not_bracket(prev_label)):
                spnsp = " "
            elif (label == "]" and is_not_bracket(prev_label)) or (is_bracket(label) and is_bracket(prev_label)) or (is_not_bracket(label) and prev_label == "["):
                spnsp = ""
            else:
                spnsp = ""

            if label =="0" or label=="\n":
                break

            text = text + spnsp + label

            prev_label = label
            i += 1

        labels.append(text)

        if len(y) == 1:
            return labels[0]

        return labels

def send_email(image, image_diff=None):
    byteArr = io.BytesIO()
    image.save(byteArr, format="PNG")
    #image_data = open(image, 'rb').read()
    msg = MIMEMultipart()
    msg['Subject'] = 'Screenshot Email'
    msg['From'] = 'Nektarios Kalogridis <mail@nektarios.com>'
    msg['To'] = 'Nektarios Kalogridis <mail@nektarios.com>'

    text = MIMEText("Here is your screenshot...")
    msg.attach(text)
    image = MIMEImage(byteArr.getvalue()) #, name=os.path.basename(image))
    msg.attach(image)

    smtp = smtplib.SMTP("smtp.nektarios.com", 587)
    smtp.ehlo()
    smtp.starttls()
    smtp.ehlo()
    smtp.login("mail@nektarios.com", "Trance5361")
    smtp.sendmail("Nektarios Kalogridis <alert@nektarios.com>", "Nektarios Kalogridis <mail@nektarios.com>", msg.as_string())
    smtp.sendmail("Nektarios Kalogridis <alert@nektarios.com>", "Nektarios Kalogridis <7324230342@tmomail.net>", msg.as_string())
    smtp.quit()


def load_data():
    df_out = pd.DataFrame()
    labels = []
    types =[]
    texts = []
    deal_names = []
    span_ids = []

    for i in range(802):
        text_path = "data/ALL_Final_3/ALL_" + str(i) + "Text.png"
        deal_name_path = "data/ALL_Final_2/ALL_" + str(i) + "_DealName.png"
        labels_path = "data/ALL_Final_3/ALL_" + str(i) + "_Label.png"
        type_path = "data/ALL_Final_3/ALL_" + str(i) + "_Type.png"
        span_id_path = "data/ALL_Final_3/ALL_" + str(i) + "_SpanId.png"

        text_image = cv2.imread(text_path)
        text_image = cv2.cvtColor(text_image, cv2.COLOR_BGR2GRAY)

        labels_image = cv2.imread(labels_path)
        labels_image = cv2.cvtColor(labels_image, cv2.COLOR_BGR2GRAY)

        type_image = cv2.imread(type_path)
        type_image = cv2.cvtColor(type_image, cv2.COLOR_BGR2GRAY)

        span_id_image = cv2.imread(span_id_path)
        span_id_image = cv2.cvtColor(span_id_image, cv2.COLOR_BGR2GRAY)

        deal_name_image = cv2.imread(deal_name_path)
        deal_name_image = cv2.cvtColor(deal_name_image, cv2.COLOR_BGR2GRAY)

        text = pytesseract.image_to_string(text_image) #.replace("\n", "").replace("eachL", "each L").replace("tothe", "to the").replace("TelearateScreen", "Telerate Screen").replace("serviceis", "service is").replace("thebasis", "the basis") #.replace(" BOR"," LIBOR").replace("!", "I").replace("LBOR", "LIBOR").replace("willbe", "will be").replace(" uch"," such").replace("tothe", "to the").replace("UBOR", "LIBOR").replace("\n", "").replace("page3750", "page 3750").replace(" i ", " in ")

        label = pytesseract.image_to_string(labels_image)
            #replace("I", "l").replace("|", "l").replace("!", "l").replace("dtr", "d t r").replace("dt","d t").replace("tt", "t t").replace("nr", "n r").\
            #replace("1", "l").replace(" -ln", " r-ln").replace("f", "[").replace("ll", "]]").replace("}", "]").replace("nyl","ny l").replace("tl", "t l").replace("ta", "t a").replace("[v]", "fv]").\
            #replace("td", "t d").replace("tr"," t r").replace("tl", "t l").replace("ld","l d").replace("[d t]", '[d t').replace("r-lnl", "r-ln l").replace("rln", "r-ln").replace("rin", "r-ln").replace("[y]", "fv]" ).replace("tl", "t l")

        label = get_label_string(label)

        type = pytesseract.image_to_string(type_image)

        deal_name = pytesseract.image_to_string(deal_name_image)
        span_id = pytesseract.image_to_string(span_id_image)

        labels.append(label)
        types.append(type)
        texts.append(text)
        deal_names.append(deal_name)
        span_ids.append(span_id)
        print(str(i)+"\t", deal_name+"\t", span_id+"\t", label+"\t", type)
        print(text)


    df_out["Deal Name"] = deal_names
    df_out["Span Id"] = span_ids
    df_out["Text"] = texts
    df_out["Sequence Labels"] = labels
    df_out["Type"] = types

    df_out.to_excel("data/ALL_Final_3.xlsx")



def screen_capture():
    time.sleep(10)

    for i in range(3922):

        screen_image_desk_name_grab = ImageGrab.grab(bbox=(798, 192, 882, 214))
        screen_image_desk_name_grab.save("data/ALL/All Data/ALL_"+str(i)+"_DeskName.png")
        keyboard.press_and_release("right")
        time.sleep(0.6)
        screen_image_deal_name_grab = ImageGrab.grab(bbox=(798, 192, 914, 214))
        screen_image_deal_name_grab.save("data/ALL/All Data/ALL_"+str(i)+"_DealName.png")
        keyboard.press_and_release("right")
        time.sleep(0.6)
        screen_image_span_id_grab = ImageGrab.grab(bbox=(798, 192, 825, 214))
        screen_image_span_id_grab.save("data/ALL/All Data/ALL_"+str(i)+"_SpanId.png")
        keyboard.press_and_release("right")
        time.sleep(0.6)
        screen_image_text_grab = ImageGrab.grab(bbox=(798, 192, 1890, 1331))
        screen_image_text_grab.save("data/ALL/All Data/ALL_"+str(i)+"_Text.png")
        keyboard.press_and_release("right")
        time.sleep(0.6)
        screen_image_sequence_label_grab = ImageGrab.grab(bbox=(798, 192, 1040, 214))
        screen_image_sequence_label_grab.save("data/ALL/All Data/ALL_"+str(i)+"_Label.png")
        keyboard.press_and_release("right")
        time.sleep(0.6)
        screen_image_type_grab = ImageGrab.grab(bbox=(798, 192, 825, 214))
        screen_image_type_grab.save("data/ALL/All Data/ALL_"+str(i)+"_Type.png")
        keyboard.press_and_release("right")
        keyboard.press_and_release("down")
        keyboard.press_and_release("home")
        time.sleep(0.6)


        #screen_image_grab = ImageGrab.grab(bbox=(644, 178, 850, 198))
        #screen_image_grab = ImageGrab.grab(bbox=(644, 178, 665, 198))

        #screen_image_grab.save("data/ALL/SpanId/ALL_" + str(i) + "_SpanId.png")



if __name__ == '__main__':
   load_data()
   #screen_capture()
