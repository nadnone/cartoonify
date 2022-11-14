import cv2
from PIL import Image

def custom_filter(image):
     for i in range(1,4):
        image = max_filter(image, i**2)
        image = min_filter(image, (i*2)**2)

        return image

def edge_mask(image):

    # Grayscale
    edge_mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # blur
    edge_mask = cv2.medianBlur(edge_mask, 9)
    edge_mask = cv2.blur(edge_mask, (32,32), 16)

    # edge detection
    edge_mask = cv2.adaptiveThreshold(
        edge_mask, 
        255, 
        cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, 
        11, 
        1
        )

    edge_mask = min_filter(edge_mask, 3)
    edge_mask = max_filter(edge_mask, 6)

    edge_mask = cv2.medianBlur(edge_mask, 3)
    edge_mask = custom_filter(edge_mask)

    edge_mask = cv2.medianBlur(edge_mask, 11)

    edge_mask = min_filter(edge_mask, 8)


    return edge_mask

def color_reduce(filename, reducer):
    
    image = Image.open(filename)

    print("[*] réduction de la palette de couleurs..")
    image = image.quantize(reducer)

    image.save("reduce.png")

    img_reduce = cv2.imread("reduce.png")

    return img_reduce


def min_filter(image, k_size):

    # kernel
    size = (k_size,k_size)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)

    # on prend le minimum
    rslt = cv2.erode(image, kernel)

    return rslt

def max_filter(image, k_size):

    # kernel
    size = (k_size,k_size)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)

    # on prend le minimum
    rslt = cv2.dilate(image, kernel)

    return rslt




# on charge l'image
original_image = cv2.imread("test.jpg")

print("[*] génération du masque de bords..")
edge_mask_img = edge_mask(original_image)
cv2.imwrite("edge.jpg", edge_mask_img)


# traitement intermédiaire
img_reduce = cv2.blur(original_image, (128,128), 50)

img_reduce = min_filter(img_reduce, 128)

img_reduce = max_filter(img_reduce, 128)

img_reduce = cv2.blur(img_reduce, (64,64), 50)

for i in range(0,3):
    img_reduce = max_filter(img_reduce, 64)
    img_reduce = min_filter(img_reduce, 64)

cv2.imwrite("blur.jpg", img_reduce)

img_reduce = color_reduce("blur.jpg", 32)
img_reduce = custom_filter(img_reduce)

# on bitwise
cartoon = cv2.bitwise_and(img_reduce, img_reduce, mask=edge_mask_img)


cv2.imwrite("cartoon.png", cartoon)