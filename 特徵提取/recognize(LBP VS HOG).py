# import the necessary packages

from localbinarypatterns import LocalBinaryPatterns
from sklearn.ensemble import RandomForestClassifier
from skimage import feature
from imutils import paths
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True, help="path to the training images")
ap.add_argument("-e", "--testing", required=True, help="path to the tesitng images")
args = vars(ap.parse_args())

# initialize the local binary patterns descriptor along with
# the data and label lists
desc = LocalBinaryPatterns(12, 24)
data = []
labels = []

# loop over the training images

for imagePath in paths.list_images(args["training"]):

        # load the image, convert it to grayscale, and describe it
        image = cv2.imread(imagePath)
        
        #LBP
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #hist = desc.describe(gray)
        
        #HOG
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        Cutted = cv2.resize(gray, (500, 500))
        #取得其HOG資訊及視覺化圖檔
        (hist, hogImage) = feature.hog(Cutted, orientations=9, pixels_per_cell=(10, 10), cells_per_block=(2, 2), transform_sqrt=True, visualize=True)
        
        # extract the label from the image path, then update the
        # label and data lists
        labels.append(imagePath.split("\\")[-2])
        data.append(hist)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(data, labels)

# loop over the testing images

for imagePath in paths.list_images(args["testing"]):

        # load the image, convert it to grayscale, describe it,
        # and classify it
        image = cv2.imread(imagePath)
        
        #LBP
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #hist = desc.describe(gray)

        #HOG
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        Cutted = cv2.resize(gray, (500, 500))
        #取得其HOG資訊及視覺化圖檔
        (hist, hogImage) = feature.hog(Cutted, orientations=9, pixels_per_cell=(10, 10), cells_per_block=(2, 2), transform_sqrt=True, visualize=True)
        
        # extract the label from the image path, then update the
        # label and data lists
        labels.append(imagePath.split("\\")[-2])
        data.append(hist)
        
        prediction = model.predict(hist.reshape(1, -1))[0]

        # display the image and the prediction
        cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
cv2.destroyAllWindows()

        