import cv2

if __name__ == '__main__':
    path = 'sample.jpg'
    img = cv2.imread(path)
    if img is None:
        print('No image to show')
    else:
        cv2.imshow('Input image', img)
        # Wait up to 5s for a keypress
        cv2.waitKey(5000);
