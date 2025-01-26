import cv2
import pytesseract
import numpy as np
import google.generativeai as genai
import pyttsx3 


engine = pyttsx3.init()


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


img = cv2.imread('dolo.jpg')


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


text_list = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    roi = img[y:y+h, x:x+w]  
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(thresh_roi, config='--psm 11').strip()
    if text:
        text_list.append(text)


extracted_text = "\n".join(text_list)
print("Extracted Text:\n", extracted_text)


with open('extracted_text.txt', 'w', encoding='utf-8') as f:
    f.write(extracted_text)


GEMINI_API_KEY = "AIzaSyBTMqDknC0qEi5xbP5LgPnBbFa6cUYJrDw"
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-pro")

try:
   
    prompt = f"Provide detailed information about: {extracted_text}"

    response = model.generate_content(prompt)

 
    gemini_response_text = response.text if response else "No valid response received."

    print("\nGemini Response:\n", gemini_response_text)

 
    with open('gemini_response.txt', 'w', encoding='utf-8') as f:
        f.write(gemini_response_text)

  
    if gemini_response_text.strip(): 
        engine.say(gemini_response_text)
        engine.runAndWait()

except Exception as e:
    print(f"Error querying Gemini API: {e}")


cv2.imshow('Original Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()