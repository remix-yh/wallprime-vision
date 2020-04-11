import io
from PIL import Image
import numpy as np
import cv2
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient

# 素因数分解したリストを返却する
def prime_factorize(n):
    a = []
    while n % 2 == 0:
        a.append(2)
        n //= 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            a.append(f)
            n //= f
        else:
            f += 2
    if n != 1:
        a.append(n)
    return a

# 指定した画像内に存在する数字を物体検出したBBOX情報を返却する
def predict(predictor, project_id, publish_iteration_name, img):
    results = predictor.detect_image(project_id, publish_iteration_name, img)
    resultlist = sorted(results.predictions, key=lambda p: p.bounding_box.left)   
    return resultlist

# 英単語の数値("zero"など)から、数値文字("0"など)に変換した値を返却する
def convert_number(number):
    num_dict= { 
        "zero":"0",
        "one":"1",
        "two":"2",
        "three":"3",
        "four":"4",
        "five":"5",
        "six":"6",
        "seven":"7",
        "eight":"8",
        "nine":"9"
        }
    return num_dict[number]

# binary imageに変換したimageを返却する
def convert_binary_image(img):
    image = Image.fromarray(img)
    png = io.BytesIO()
    image.save(png, format='png')
    binary_image = png.getvalue()

    return binary_image

# Main関数
def main():
    # 定数
    PUBLISH_ITERATION_NAME = '<Iteration Name>'
    PREDICTION_KEY = '<Prediction Key>'
    ENDPOINT = '<End Point>'
    PROJECT_ID = '<Project ID>'
    PROBABILITY_THRESH = 0.55

    predictor = CustomVisionPredictionClient(PREDICTION_KEY, endpoint=ENDPOINT)
    cap = cv2.VideoCapture(1)

    while(True):
        ret, frame = cap.read()
        cv2.imshow('movie',frame)

        # 64bitの場合は、0xFFを付けて&する
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            # binaryの形式に変換
            binary_image = convert_binary_image(frame)

            # 推論実施
            predictions = predict(predictor, PROJECT_ID, PUBLISH_ITERATION_NAME, binary_image)
            
            num_string = ""
            
            for prediction in predictions:
                if prediction.probability >= PROBABILITY_THRESH:
                    height,width,depth = frame.shape
                    x1 = prediction.bounding_box.left * width
                    y1 = prediction.bounding_box.top * height
                    x2 = x1 + width * prediction.bounding_box.width
                    y2 = y1 + height * prediction.bounding_box.height
                    num_string += convert_number(prediction.tag_name)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0))
            
            # 検出した数字の描画
            cv2.putText(frame, "num:" + num_string, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
            
            # 素因数の文字列生成
            prime_factor_string = "prime factor:"
            prime_factor_list = prime_factorize(int(num_string))
            for prime_factor in prime_factor_list:
                prime_factor_string += str(prime_factor)
                prime_factor_string += " "
            
            # 素因数の描画
            cv2.putText(frame, prime_factor_string, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
            cv2.imshow("object detection", frame)
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()