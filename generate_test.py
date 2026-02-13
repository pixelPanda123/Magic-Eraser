import cv2
import numpy as np

# 1. Create a 512x512 green background (The "Field")
img = np.zeros((512, 512, 3), dtype=np.uint8)
img[:] = (34, 139, 34) # Forest Green

# 2. Draw a Red Circle (The "Object" to remove)
cv2.circle(img, (256, 256), 50, (0, 0, 255), -1)
cv2.imwrite("test_image.png", img)

# 3. Create a Binary Mask (White where the object is, Black elsewhere)
mask = np.zeros((512, 512), dtype=np.uint8)
cv2.circle(mask, (256, 256), 55, 255, -1) # Slightly larger to ensure clean edges
cv2.imwrite("test_mask.png", mask)

print("✅ test_image.png and test_mask.png created at 512x512 resolution.")