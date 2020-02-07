from base.base_data_preprocess import BasePreprocess
import cv2
import numpy as np

class KagglePreprocess(BasePreprocess):
    def __init__(self):
        super(BasePreprocess, self).__init__()
        
    def clahe(self, img, clipLimit=2.0, gridsize=8):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clipLimit,tileGridSize=(gridsize,gridsize))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return img

    def subtract_gaussian_blur(self, img,b=5):
        gb_img = cv2.GaussianBlur(img, (0, 0), b)
        return cv2.addWeighted(img, 4, gb_img, -4, 128)

    def remove_outer_circle(self, a, p, r):
        b = np.zeros(a.shape, dtype=a.dtype)
        cv2.circle(b, (a.shape[1] // 2, a.shape[0] // 2), int(r * p), (1, 1, 1), -1, 8, 0)
        return a * b + 128 * (1 - b)

    def crop_img(self, img):
        non_zeros = img.nonzero() # Find indices of non zero elements
        non_zero_rows = [min(np.unique(non_zeros[0])), max(np.unique(non_zeros[0]))] # Find the first and last row with non zero elements
        non_zero_cols = [min(np.unique(non_zeros[1])), max(np.unique(non_zeros[1]))] # Find the first and last row with non zero elements
        crop_img = img[non_zero_rows[0]:non_zero_rows[1], non_zero_cols[0]:non_zero_cols[1],:] # Crop the image
        return crop_img

    def make_square(self, img, min_size=256):
        x, y, z = img.shape
        size = max(min_size, x, y)
        new_img = np.zeros((size,size,z), dtype=img.dtype)
        for i in range(z):
            new_img[:,:,i] = img[0,0,i]
        new_img[int((size-x)/2):int((size-x)/2+x), int((size-y)/2):int((size-y)/2+y),:] = img
        return np.array(new_img)

    def preprocess(self, img, numOfBlurred):
        dtype = img.dtype
        if img is not None:
            img = self.crop_img(img)
            blurred_imgs = np.zeros([img.shape[0],img.shape[1],numOfBlurred], dtype=dtype)
            for row in range(numOfBlurred):
                img2 = self.subtract_gaussian_blur(img,(row+1)*5)
                blurred_imgs[:,:,row] = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            img = self.make_square(blurred_imgs)
            img = self.remove_outer_circle(img, 0.97, img.shape[0]//2)
            return img
        else:
            raise ValueError