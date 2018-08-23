
# Import the modules
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np


# Load the SVM model
clf, pp = joblib.load("multi_digits_svm.pkl")


def main():  
    # Open Camera
    cap = cv2.VideoCapture(0)

    while (cap.isOpened()):
        # Capture frames from the camera
        ret, img = cap.read()
        # Apply get_img_contour_thresh function on frame
        img, contours, thresh = get_img_contour_thresh(img)
        
        ans = ''
        
        if len(contours) > 0:
            for ctr in contours:
                # Ranging contourArea
                if cv2.contourArea(ctr) > 1500 and cv2.contourArea(ctr)<5000 :
                    # Get rectangles contains each contour i.e. digit
                    rect = cv2.boundingRect(ctr)
                    # Dimensions of rectangle
                    x, y, w, h = rect
                
                    # Making new image containing coutour for classification
                    newImage = thresh[y:y + h, x:x + w]
                    # Resize the image
                    newImage = cv2.resize(newImage, (28, 28))
                    # Calculate the HOG features
                    newImage = np.array(newImage)
                    hog_ft = hog(newImage, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
                    # Perform classification
                    hog_ft = pp.transform(np.array([hog_ft], 'float64'))
                    ans = clf.predict(hog_ft)
                    # Make the rectangular region around the digit and texting classified digit
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 4)
                    cv2.putText(img, str(int(ans[0])), (x, y),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
               
        #Showing frame and threshold
        cv2.imshow("Frame", img)
        cv2.imshow("Contours", thresh)
        k = cv2.waitKey(10)
        if k == 13:
            break
            
def get_img_contour_thresh(img):
    
    # Change color-space from BGR -> Gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur and Threshold
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    _,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    return img, contours, thresh

main()

