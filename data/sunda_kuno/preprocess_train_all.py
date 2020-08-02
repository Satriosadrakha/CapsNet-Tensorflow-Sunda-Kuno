# from PIL import Image
# from numpy import asarray
# import numpy as np

# # load the image
# image = Image.open('HA_3.png')
# # convert image to numpy array
# data = asarray(image)
# print(type(data))
# # summarize shape
# print(data.shape)

# # create Pillow image
# image2 = Image.fromarray(data)
# print(type(image2))

# # summarize image details
# print(image2.mode)
# print(image2.size)

# im = np.array(Image.open('HA_3.png').convert('L').resize((30,30))) #you can pass multiple arguments in single line
# print(type(im))

# Image.fromarray(im).save('HA_3_transformed.png')

# print(im)

import cv2 as cv
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import PIL.ImageOps
import os

def resize_only(file_name):
	img = Image.open(file_name).convert('LA')
	# img = img.filter(ImageFilter.SMOOTH)
	# img = img.filter(ImageFilter.SHARPEN)
	img = img.resize((28, 28))
	# img = PIL.ImageOps.invert(img)
	img.save("temp.png")
	img = cv.imread('temp.png', 0)
	img = cv.bitwise_not(img)
	img =  Image.fromarray(img)
	
	return img

def resize(file_name, invert):
	# image = cv.imread('%s' % file_name,0)
	img = Image.open(file_name).convert('LA')
# 	if invert==True:
# 		img = PIL.ImageOps.invert(img)
	enhancer = ImageEnhance.Contrast(img)
	
	factor = 2.0
	img = enhancer.enhance(factor)
	img = img.filter(ImageFilter.SMOOTH)
	img = img.filter(ImageFilter.SHARPEN)
	img = img.resize((28, 28))
	# print(type(img))    
    
	img.save("temp.png")
    
	img = cv.imread('temp.png', 0)
    
	if invert==True:
		img = cv.bitwise_not(img)
	ret,th = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    
	blur = cv.GaussianBlur(img,(5,5),0)
	ret2,th2 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    
	mask = np.where(th ==  0, th2, 255)
    
	img = np.where(mask == 0, img, 255)

    # create a CLAHE object (Arguments are optional).
	clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	img = clahe.apply(img)

	_,img_remove = cv.threshold(img,95,255,cv.THRESH_TOZERO)
	img_remove = cv.bitwise_not(img_remove)
	_,img = cv.threshold(img_remove,95,255,cv.THRESH_TOZERO)

	# img = cv.bitwise_not(img)
    
	img =  Image.fromarray(img)
    
	# print(img)
	# pix = np.array(img)
	# print(type(pix))
	# print(pix.shape)
	# print(pix)
	# pix = pix[:, :, 0]
	# print(type(pix))
	# print(pix.shape)
	# print(pix)
	return img
	# img.save("resized_"+file_name)

def main():
	# crop(str(sys.argv))
	script_dir = os.path.dirname(__file__)
	i=["A","BA","CA","DA","GA","HA","I","JA","KA","LA","MA","NA","NGA","NYA","PA","PANELENG","PANEULEUNG","PANGHULU","PANGLAYAR","PANOLONG","PANYUKU","PATEN","RA","SA","TA","U","WA","YA"]
	j=[30,47,19,67,37,27,16,21,60,60,56,120,25,14,61,42,63,60,23,35,36,84,56,90,78,18,24,22]
	k=[12,30,7,45,16,11,7,9,40,25,24,80,10,6,39,28,42,40,10,24,24,36,36,60,52,7,10,9]
	ganed = [0,2,4,5,6,7,12,13,18,19,20,25,26,27]
    
#   train_image to ready_to_train
	sumImg = 0
	for x in range(0, len(i), 1):
		sumImg = sumImg + j[x]
		for y in range(1, j[x]+1):
			rel_path = "train_image/%s_%s.png" % (i[x],str(y))
# 			rel_path = "ready_to_train/%s_%s.png" % (i[x],str(y))
			abs_file_path = os.path.join(script_dir, rel_path)
            
			img = resize_only(abs_file_path)
			img.save("ready_to_train/%s_%s.png" % (i[x],str(y)))
			# img = resize(abs_file_path, invert=False)
			# img.save("ready_to_train_3/%s_%s.png" % (i[x],str(y)))

# #   GAN_image to ready_to_train
# 	for x in ganed:
# 		for y in range(1, 26):
# 			rel_path = "GAN_generated_images/%s_%s.png" % (i[x],str(y+j[x]))
# # 			rel_path = "ready_to_train/%s_%s.png" % (i[x],str(y))
# 			abs_file_path = os.path.join(script_dir, rel_path)
            
# 			img = resize(abs_file_path, invert=True)
# 			img.save("ready_to_train_3/%s_%s.png" % (i[x],str(y+j[x])))

#   test_image to ready_to_test (validation)
	sumImg = 0
	final_np = np.array([])
	final_label = np.array([])
	for x in range(0, len(i), 1):
		sumImg = sumImg + k[x]
		for y in range(1, int((k[x])/2)+1):
			rel_path = "test_image/%s/%s_%s.png" % (i[x],i[x],str(y))
			# rel_path = "ready_to_train/%s_%s.png" % (i[x],str(y))
			abs_file_path = os.path.join(script_dir, rel_path)

			# img = resize(abs_file_path, invert=False)
			# img.save("ready_to_test_2/%s_%s.png" % (i[x],str(y)))
			img = resize_only(abs_file_path)
			img.save("ready_to_test/%s_%s.png" % (i[x],str(y)))

# 	test_image to ready_to_test (test)
	sumImg = 0
	for x in range(0, len(i), 1):
		sumImg = sumImg - (-k[x]//2)
		for y in range((-(-k[x])//2)+1, k[x]+1):
#             rel_path = "test_image/%s/%s_%s.png" % (i[x],i[x],str(y))
			rel_path = "test_image/%s/%s_%s.png" % (i[x],i[x],str(y))
			# rel_path = "ready_to_train/%s_%s.png" % (i[x],str(y))
			abs_file_path = os.path.join(script_dir, rel_path)
            
			img = Image.open(abs_file_path)
			img = np.array(img)
			img = img[:, :, 0]
			img = resize_only(abs_file_path)
			img.save("ready_to_test/%s_%s.png" % (i[x],str(y)))
			# img = resize(abs_file_path, invert=False)
			# img.save("ready_to_test_2/%s_%s.png" % (i[x],str(y)))

if __name__ == "__main__":
	main()

