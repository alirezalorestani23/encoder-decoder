# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import random

import cv2
import numpy as np


def inverse_zigzag(input, vmax, hmax):
    # print input.shape

    # initializing the variables
    # ----------------------------------
    h = 0
    v = 0

    vmin = 0
    hmin = 0

    output = np.zeros((vmax, hmax))

    i = 0
    # ----------------------------------

    while ((v < vmax) and (h < hmax)):
        # print ('v:',v,', h:',h,', i:',i)
        if ((h + v) % 2) == 0:  # going up

            if (v == vmin):
                # print(1)

                output[v, h] = input[i]  # if we got to the first line

                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1

                i = i + 1

            elif ((h == hmax - 1) and (v < vmax)):  # if we got to the last column
                # print(2)
                output[v, h] = input[i]
                v = v + 1
                i = i + 1

            elif ((v > vmin) and (h < hmax - 1)):  # all other cases
                # print(3)
                output[v, h] = input[i]
                v = v - 1
                h = h + 1
                i = i + 1


        else:  # going down

            if ((v == vmax - 1) and (h <= hmax - 1)):  # if we got to the last line
                # print(4)
                output[v, h] = input[i]
                h = h + 1
                i = i + 1

            elif (h == hmin):  # if we got to the first column
                # print(5)
                output[v, h] = input[i]
                if (v == vmax - 1):
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1

            elif ((v < vmax - 1) and (h > hmin)):  # all other cases
                output[v, h] = input[i]
                v = v + 1
                h = h - 1
                i = i + 1

        if ((v == vmax - 1) and (h == hmax - 1)):  # bottom right element
            # print(7)
            output[v, h] = input[i]
            break

    return output


def decode_frames(i, block_size):
    quantize_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                [12, 12, 14, 19, 26, 58, 60, 55],
                                [14, 13, 16, 24, 40, 57, 69, 56],
                                [14, 17, 22, 29, 51, 87, 80, 62],
                                [18, 22, 37, 56, 68, 109, 103, 77],
                                [24, 35, 55, 64, 81, 104, 113, 92],
                                [49, 64, 78, 87, 103, 121, 120, 101],
                                [72, 92, 95, 98, 112, 100, 103, 99]])
    with open("encoded_image" + str(i) + ".txt", 'r') as myfile:
        encode_frame = myfile.read()

    details = encode_frame.split()
    r = int(details[0])
    c = int(details[1])

    array = np.zeros(r * c).astype(int)

    # some loop var initialisation
    array_index = 0
    details_index = 2
    zeros_counter = 0

    # This loop gives us reconstructed array of size of image

    while array_index < array.shape[0]:
        # Oh! image has ended
        if details[details_index] == ';':
            break
        array[array_index] = int(details[details_index])

        if details_index + 3 < len(details):
            zeros_counter = int(details[details_index + 3])

        if zeros_counter == 0:
            array_index = array_index + 1
        else:
            array_index = array_index + zeros_counter + 1

        details_index = details_index + 2

    array = np.reshape(array, (r, c))

    reshaped_array_rows = 0

    # initialisation of compressed image
    reconstructed_frame = np.zeros((r, c), dtype=np.uint8)

    while reshaped_array_rows < r:
        reshaped_array_columns = 0
        while reshaped_array_columns < c:
            temp_stream = array[reshaped_array_rows:reshaped_array_rows + 8,
                          reshaped_array_columns: reshaped_array_columns + 8]
            block = inverse_zigzag(temp_stream.flatten(), int(block_size), int(block_size))
            de_quantized = np.multiply(block, quantize_matrix)
            reconstructed_frame[reshaped_array_rows:reshaped_array_rows + 8,
            reshaped_array_columns:reshaped_array_columns + 8] = cv2.idct(de_quantized)
            reshaped_array_columns = reshaped_array_columns + 8
        reshaped_array_rows = reshaped_array_rows + 8

    # clamping to  8-bit max-min values
    reconstructed_frame[reconstructed_frame > 255] = 255
    reconstructed_frame[reconstructed_frame < 0] = 0

    print("decode: ", i)

    return reconstructed_frame


def encode_frames(i, frame, block_size):
    quantize_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                [12, 12, 14, 19, 26, 58, 60, 55],
                                [14, 13, 16, 24, 40, 57, 69, 56],
                                [14, 17, 22, 29, 51, 87, 80, 62],
                                [18, 22, 37, 56, 68, 109, 103, 77],
                                [24, 35, 55, 64, 81, 104, 113, 92],
                                [49, 64, 78, 87, 103, 121, 120, 101],
                                [72, 92, 95, 98, 112, 100, 103, 99]])
    r, c = np.shape(frame)
    r1, c1 = int(r / block_size) * block_size, int(c / block_size) * block_size
    new_frame = np.zeros((r1, c1))
    new_frame[0:r1, 0:c1] = frame[0:r1, 0:c1]
    for k in range(0, r - block_size, block_size):
        for j in range(0, c - block_size, block_size):
            block = new_frame[k:k + block_size, j:j + block_size]
            dct_frame = cv2.dct(block)
            quantized_frame = np.divide(dct_frame, quantize_matrix).astype(int)
            zigzag_scanned_frame = zigzag(quantized_frame)
            new_frame[k:k + block_size, j:j + block_size] = np.reshape(zigzag_scanned_frame, (block_size, block_size))

    encoded_frame = run_length_encoding(new_frame.flatten())
    bitstream = str(new_frame.shape[0]) + " " + str(new_frame.shape[1]) + " " + encoded_frame + ";"
    file = open("encoded_image" + str(i) + ".txt", "w")
    file.write(bitstream)
    file.close()
    print("encode: ", i)
    return r1, c1


def zigzag(input):
    # initializing the variables
    # ----------------------------------
    h = 0
    v = 0

    vmin = 0
    hmin = 0

    vmax = input.shape[0]
    hmax = input.shape[1]

    # print(vmax ,hmax )

    i = 0

    output = np.zeros((vmax * hmax))
    # ----------------------------------

    while ((v < vmax) and (h < hmax)):

        if ((h + v) % 2) == 0:  # going up

            if (v == vmin):
                # print(1)
                output[i] = input[v, h]  # if we got to the first line

                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1

                i = i + 1

            elif ((h == hmax - 1) and (v < vmax)):  # if we got to the last column
                # print(2)
                output[i] = input[v, h]
                v = v + 1
                i = i + 1

            elif ((v > vmin) and (h < hmax - 1)):  # all other cases
                # print(3)
                output[i] = input[v, h]
                v = v - 1
                h = h + 1
                i = i + 1


        else:  # going down

            if ((v == vmax - 1) and (h <= hmax - 1)):  # if we got to the last line
                # print(4)
                output[i] = input[v, h]
                h = h + 1
                i = i + 1

            elif (h == hmin):  # if we got to the first column
                # print(5)
                output[i] = input[v, h]

                if (v == vmax - 1):
                    h = h + 1
                else:
                    v = v + 1

                i = i + 1

            elif ((v < vmax - 1) and (h > hmin)):  # all other cases
                # print(6)
                output[i] = input[v, h]
                v = v + 1
                h = h - 1
                i = i + 1

        if ((v == vmax - 1) and (h == hmax - 1)):  # bottom right element
            # print(7)
            output[i] = input[v, h]
            break

    # print ('v:',v,', h:',h,', i:',i)
    return output


def run_length_encoding(zigzag_scanned_frame):
    i = 0
    skip = 0
    stream = []
    bitstream = ""
    zigzag_scanned_frame = zigzag_scanned_frame.astype(int)
    while i < zigzag_scanned_frame.shape[0]:
        if zigzag_scanned_frame[i] != 0:
            stream.append((zigzag_scanned_frame[i], skip))
            bitstream = bitstream + str(zigzag_scanned_frame[i]) + " " + str(skip) + " "
            skip = 0
        else:
            skip = skip + 1
        i = i + 1

    return bitstream


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    block_size = 8

    # reshaped = np.reshape(zigzag(quantize_matrix), (8, 8))
    # print(run_length_encoding(reshaped.flatten()))

    h, w = 536, 960
    cap = cv2.VideoCapture("a.avi")
    i = 50
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     h, w = encode_frames(i, frame, block_size)
    #     i += 1

    j = 0
    frameSize = (h, w)
    out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize)

    while j < i:
        out.write(decode_frames(j, block_size))
        j += 1
    out.release()

    cap.release()
    cv2.destroyAllWindows()
