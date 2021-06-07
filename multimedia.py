# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2


def encode_frames(i,frame):
    print(str(i)+"hi")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cap = cv2.VideoCapture("a.avi")
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        encode_frames(i,frame)
        # cv2.imwrite('kang' + str(i) + '.jpg', frame)
        i += 1

    cap.release()
    cv2.destroyAllWindows()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
