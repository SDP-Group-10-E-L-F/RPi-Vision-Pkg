import numpy as np
from PIL import Image
import tensorflow as tf
import mediapipe as mp
import imutils
import cv2

class_names = ['Empty','Longsleeve', 'Pants', 'T-Shirt']

batch_size = 32
img_height = 400
img_width = 400

TF_MODEL_FILE_PATH = 'TFlite_Models/model0.tflite'
interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)
print(f"Signature list : {interpreter.get_signature_list()}")

input_details = interpreter.get_input_details()
print(f"Input details : {input_details}")

output_details = interpreter.get_output_details()
print(f"Output details : {output_details}")
interpreter.allocate_tensors()

def predict(model, frame):
    predictions = model.predict(frame)
    score = tf.nn.softmax(predictions[0])
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    return score

"""
Hand Detection Part
"""
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=4)
mpDraw = mp.solutions.drawing_utils

# Processing the input image
def process_image(img):
    # Converting the input to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(gray_image)

    # Returning the detected hands to calling function
    return results

def draw_bounding_box(img, results):
    """
    Args:
        img: <class 'numpy.ndarray'>
        results:

    Returns:
    """
    if results.multi_hand_landmarks:
        for hand_landmark, hand_classification in zip(results.multi_hand_landmarks, results.multi_handedness):
            img_height, img_width, _ = img.shape
            x = [int(landmark.x * img_width) for landmark in hand_landmark.landmark]
            y = [int(landmark.y * img_height) for landmark in hand_landmark.landmark]
            score = np.mean([float(classification.score) for classification in hand_classification.classification])
            score = "{:.2f}".format(round(score, 2))

            left = np.min(x)
            right = np.max(x)
            bottom = np.min(y)
            top = np.max(y)

            thick = int((img_height + img_width) // 400)

            line_width = max(round(sum(img.shape) / 2 * 0.003), 2)  # line width

            # Bouding box visualization
            cv2.rectangle(img,
                          (left - 10, top + 10),    # Top left coordinates
                          (right + 10, bottom - 10),    # Bottom right coordinates
                          (255, 0, 0),  # Color of the detection box
                          thickness=line_width,
                          lineType=cv2.LINE_AA)

            # Text info display on bounding box
            tf = max(line_width - 1, 1)  # font thickness

            # text width, height
            w, h = cv2.getTextSize(f'Hand {score}', 0, fontScale=line_width / 3, thickness=tf)[0]
            outside = (left - 10) - h >= 3
            p2 = (left - 10) + w, (top + 10) - h - 3 if outside else (top + 10) + h + 3
            cv2.rectangle(img, (left - 10, top + 10), p2, (255, 0, 0), -1, cv2.LINE_AA)  # filled
            cv2.putText(img,
                        f'Hand {score}', ((left - 10), (top + 10) - 2 if outside else (top + 10) + h + 2),
                        0,
                        line_width / 3,
                        (255, 255, 255),
                        thickness=tf,
                        lineType=cv2.LINE_AA)

def is_hand_detected(results):
    if results.multi_hand_landmarks and results.multi_handedness:
        print("Hands Detected! Stop Folding")
        return True
    else:
        print("Keep Folding")
        return False

def return_clothe_type(frame):
    interpreter.set_tensor(input_details[0]['index'], frame)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    score = tf.nn.softmax(predictions[0])
    cloth_type = class_names[np.argmax(score)]
    return cloth_type, score

def arr2img(frame, width=400, height=400):
    """
    Args:
        frame: numpy.ndarray
    """
    frame = cv2.resize(frame, (width, height))
    img = Image.fromarray(frame, 'RGB')
    img.save('my.png')
    img.show()


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('cannot open the camera')
        exit()
    while cap.isOpened():
        ret, frame, = cap.read()
        cloth_frame = frame
        hand_frame = frame
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # For clothes detection
        cloth_frame = cv2.resize(cloth_frame, (400, 400))
        cloth_frame_arr = tf.expand_dims(tf.keras.utils.img_to_array(cloth_frame), 0)
        cloth_type, score = return_clothe_type(cloth_frame_arr)

        print(
            "This image most likely belongs to {} with a {:.3f} percent confidence."
            .format(cloth_type, round(np.max(score), 3))
        )

        cv2.putText(cloth_frame, f'{cloth_type} : {round(np.max(score), 3)}',
                    (0, img_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA)

        # For hands detection
        hand_frame = imutils.resize(hand_frame, width=img_width, height=img_height)
        results = process_image(hand_frame)
        draw_bounding_box(hand_frame, results)

        is_hand_detected(results)

        parallel = np.concatenate((cloth_frame, hand_frame), axis=0)

        cv2.imshow('Classification', parallel)

        # Program terminates when q key is pressed
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()

