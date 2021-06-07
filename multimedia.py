# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import numpy as np


def encode_frames(i, frame):
    
    q_mtx = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                      [12, 12, 14, 19, 26, 58, 60, 55],
                      [14, 13, 16, 24, 40, 57, 69, 56],
                      [14, 17, 22, 29, 51, 87, 80, 62],
                      [18, 22, 37, 56, 68, 109, 103, 77],
                      [24, 35, 55, 64, 81, 104, 113, 92],
                      [49, 64, 78, 87, 103, 121, 120, 101],
                      [72, 92, 95, 98, 112, 100, 103, 99]], np.int32)
    c, r = np.shape(frame)
    a = np.zeros(shape=(c,r), dtype=np.int32)
    print(frame)
    print("------------------------------------------------------------\n\n\n\n")
    for k in range(0, c-8, 8):
        for j in range(0, r-8, 8):
            imf = np.float32(frame[ k:k + 8, j:j + 8]) / 255.0
            dct = cv2.dct(imf)
            imgcv1 = np.uint8(dct * 255.0)
            a[k:k + 8, j:j + 8] = imgcv1/q_mtx
    print(a)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cap = cv2.VideoCapture("a.avi")
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        encode_frames(i, frame)
        
        # cv2.imwrite('kang' + str(i) + '.jpg', frame)
        i += 1

    cap.release()
    cv2.destroyAllWindows()
