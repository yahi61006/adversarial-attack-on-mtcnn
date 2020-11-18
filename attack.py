import tensorflow as tf
import numpy as np
import mtcnn
from mtcnn.network.factory import NetworkFactory
import matplotlib.pyplot as plt
import cv2
import pkg_resources
import matplotlib.patches as mpatches
import os

# keras_scratch_graph problem
# gpu version
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

detector = mtcnn.MTCNN()


def scale_image(image, scale):
    height, width, _ = image.shape
    width_scaled = int(np.ceil(width * scale))
    height_scaled = int(np.ceil(height * scale))
    scaleimage = cv2.resize(
        image, (width_scaled, height_scaled), interpolation=cv2.INTER_AREA)
    return scaleimage


def loss_object(label, predict_box):
    # MSE loss
    loss = tf.math.reduce_mean(tf.math.square(tf.subtract(label, predict_box)))
    return loss


def createPnet():
    weight_file = 'mtcnn_weights.npy'
    weights = np.load(weight_file, allow_pickle=True).tolist()
    pnet = NetworkFactory().build_pnet()
    pnet.set_weights(weights['pnet'])
    return pnet


pnet_attacked = createPnet()


def imageChangeToFitPnet(image, scale):
    image = scale_image(image, scale)
    image = tf.cast(image, dtype=tf.float32)
    # Normalize image
    image = (image - 127.5) * 0.0078125
    image = image[tf.newaxis, ...]
    image = tf.transpose(image, (0, 2, 1, 3))
    return image


def createLabel(image, scale):
    image = imageChangeToFitPnet(image, scale)
    label = pnet_attacked(image)
    return label[0][0, :, :, :]


def create_adversarial_pattern(image, label):
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = pnet_attacked(image)[0][0, :, :, :]
        loss = loss_object(label, prediction)
        print(loss)
    # Get the gradients of the loss to the input image
    gradient = tape.gradient(loss, image)
    # Normalization, all gradient divide the max gradient's absolute value
    norm_gradient = tf.math.divide(
        gradient, tf.math.reduce_max(tf.math.abs(gradient)))
    # print(type(gradient))
    return norm_gradient, loss


def createMask(results, image):
    mask = np.zeros((image.shape[0], image.shape[1]))

    b = results[0]  # first bounding box
    x1, y1, width, height = b['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    left_eye_y = b['keypoints']['left_eye'][1]
    right_eye_y = b['keypoints']['right_eye'][1]
    left_mouth_y = b['keypoints']['mouth_left'][1]
    right_mouth_y = b['keypoints']['mouth_right'][1]
    nose_y = b['keypoints']['nose'][1]

    # Patch coordinate
    mask_x1 = x1 + int(width/20)
    mask_x2 = x2 - int(width/20)
    mask_y1 = nose_y
    mask_y2 = y2

    if mask_y1 < 0:
        mask_y1 = 0
    if mask_x1 < 0:
        mask_x1 = 0

    mask[mask_y1:mask_y2, mask_x1:mask_x2] = 1
    # frame_patch
    # mask[y1:y2, x1:x2] = 0

    mask = np.repeat(mask[..., np.newaxis], 3, 2)
    return mask, mask_x1, mask_x2, mask_y1, mask_y2


def getPerturbationRGB(image, imageWithPerturbations):
    temp = tf.squeeze(imageWithPerturbations)
    temp = temp / 0.0078125 + 127.5
    temp = tf.transpose(temp, (1, 0, 2))
    temp = cv2.resize(temp.numpy(), (image.shape[1], image.shape[0]))

    return temp * mask


def iterative_attack(adv_image, label, mask, scale, grayscale, learning_rate, scaleNum):

    upperBound = (255 - 127.5) * 0.0078125
    lowerBound = (0 - 127.5) * 0.0078125

    my_mask = scale_image(mask, scale)
    my_mask = tf.cast(my_mask, dtype=tf.float32)

    my_mask = my_mask[tf.newaxis, ...]
    my_mask = tf.transpose(my_mask, (0, 2, 1, 3))

    maxloss = 0.0
    count = 0

    for epoch in range(10):
        tf.dtypes.cast(adv_image, tf.int32)
        tf.dtypes.cast(adv_image, tf.float32)

        perturbations, loss = create_adversarial_pattern(adv_image, label)
        perturbations = perturbations.numpy()

        if grayscale == True:
            perturbations[:, :, 0] = perturbations[:, :, 1] = perturbations[:, :, 2] = (
                perturbations[:, :, 0] + perturbations[:, :, 1] + perturbations[:, :, 2]) // 3

        adv_image = adv_image + \
            learning_rate[scaleNum] * (perturbations * my_mask)
        adv_image = tf.where(adv_image < lowerBound, lowerBound, adv_image)
        adv_image = tf.where(adv_image > upperBound, upperBound, adv_image)

        # Optimizer
        if maxloss < loss:
            maxloss = loss
            count = 0
        else:
            count += 1
        if count > 2:
            if learning_rate[scaleNum] > 0.01:
                learning_rate[scaleNum] /= 2
            count = 0

    return adv_image


# One meter to five meter
for distance in range(1, 6):
    allFileList = os.listdir('./patch_img')
    pictureList = os.listdir('./picture/{}M/normal'.format(distance))
    true_box_info = np.load('./picture/{}M/normal/info.npy'.format(distance), allow_pickle=True)
    grayscale = False

    for file in allFileList:
        patch_name = file.split('.')[0]

        for pic in pictureList:

            pic_name = pic.split('.')[0]
            if pic_name == 'info':
                break
            which_pic = int(pic_name[0]) - 1

            image = tf.keras.preprocessing.image.load_img('./picture/{}M/normal/'.format(distance) + pic)
            patch = tf.keras.preprocessing.image.load_img('patch_img/'+file)

            image = tf.keras.preprocessing.image.img_to_array(image)
            patch = tf.keras.preprocessing.image.img_to_array(patch)

            if grayscale == True:
                patch[:, :, 0] = patch[:, :, 1] = patch[:, :, 2] = (
                    patch[:, :, 0] + patch[:, :, 1] + patch[:, :, 2]) // 3

            result = detector.detect_faces(image)

            mask, mask_x1, mask_x2, mask_y1, mask_y2 = createMask(result, image)
            
            # Add code to mtcnn.py to output all scales
            # Pack them to 'scales.npy'
            scales = np.load('scales.npy')

            image2 = image.copy()

            patch = cv2.resize(patch, (abs(mask_x1-mask_x2), abs(mask_y1-mask_y2)))

            image2[mask_y1:mask_y2, mask_x1:mask_x2] = patch
            # frame_patch
            '''x1, y1, width, height = result[0]['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            image2[y1:y2,x1:x2] = image[y1:y2,x1:x2]'''
            ##########

            show_img = image2
            # 1M scales[5:8]
            # 2M scales[4:7]
            # 3M scales[2:5]
            # 4M scales[1:4]
            # 5M scales[0:3]
            scaleStartIndex = [5, 4, 2, 1, 0]
            learning_rate = [1.0, 1.0, 1.0]

            for i in range(10):
                # Pick three scale to train for each distance
                for scaleNum in range(3):
                    print('start')
                    which_scale = scaleStartIndex[distance - 1] + scaleNum

                    scale = scales[which_scale]
                    # scale = 1

                    tempImg = show_img - show_img * mask

                    adv_image = imageChangeToFitPnet(show_img, scale)

                    label = createLabel(image, scale)

                    perturbations = iterative_attack(adv_image, label, mask, scale, grayscale, learning_rate, scaleNum)

                    perturbationsRGB = getPerturbationRGB(image, perturbations.numpy())

                    if grayscale == True:
                        perturbationsRGB[:, :, 0] = perturbationsRGB[:, :, 1] = perturbationsRGB[:, :, 2] = (
                            perturbationsRGB[:, :, 0] + perturbationsRGB[:, :, 1] + perturbationsRGB[:, :, 2]) // 3

                    show_img = tempImg + perturbationsRGB

                    show_img = np.where(show_img < 0, 0, show_img)
                    show_img = np.where(show_img > 255, 255, show_img)

            image3 = show_img
            image3 = image3.astype(np.int32)
            image3 = image3.astype(np.float32)
            results = detector.detect_faces(image3)

            storePatch = image3[mask_y1:mask_y2, mask_x1:mask_x2]

            storePatch = storePatch.astype(np.int32)
            storePatch = storePatch.astype(np.float32)

            plt.figure(figsize=(image3.shape[1]*6//image3.shape[0], 6))
            plt.imshow(image3/255)
            plt.axis(False)
            ax = plt.gca()

            # Count iou
            iou_mat = np.zeros((image.shape[0], image.shape[1]))
            t_x1, t_y1, t_x2, t_y2 = true_box_info[which_pic]['box']
            iou_mat[t_y1:t_y2, t_x1:t_x2] += 1

            for b in results:
                x1, y1, width, height = b['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                if b == results[0]:
                    iou_mat[y1:y2, x1:x2] += 1

                plt.text(x1, y1, '{:.2f}'.format(b['confidence']), color='red')
                ax.add_patch(mpatches.Rectangle((x1, y1), width, height, ec='red', alpha=1, fill=None))

            iou = round(len(iou_mat[np.where(iou_mat == 2)]) /len(iou_mat[np.where(iou_mat > 0)]), 2)

            plt.savefig('result/image/rgb/mouth/normal/{}M/'.format(distance) + patch_name+'_' + pic_name + '_iou=' + str(iou)+'.jpg')

            cv2.imwrite('result/patch/rgb/mouth/normal/{}M/'.format(distance) + patch_name+'_' + pic_name + '.jpg', storePatch[:, :, [2, 1, 0]])