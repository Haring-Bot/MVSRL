import cv2
import numpy as np

import readData
import pca
import KNN

#download Data
readData.main()

#perform pca
pca.main()

#perform KNN
KNN.main()

# Create a black image
image = np.zeros((500,500,3), np.uint8)

# Specify the font and draw the text
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image, 'Hello, World!', (50, 250), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

# Display the image
cv2.imshow("Hello, World!", image)

# Wait for a key press and then close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()
