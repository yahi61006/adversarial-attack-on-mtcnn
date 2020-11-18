import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

plt.figure()

recall = [i/10 for i in range(11)]

# ------------------------------------------normal------------------------------------------------------------------------
# 全為1
'''normal_percision = [1 for i in range(11)]
normal_label = []
for i in range(1, 6):
    normal_label.append('normal_{0}M(AP={1})'.format(
        i, round(sum(normal_percision)/11, 2)))

for i in range(len(normal_label)):
    plt.plot(recall, normal_percision, 's-', label=normal_label[i])

plt.xticks(np.linspace(0, 1, 11))
plt.yticks(np.linspace(0, 1, 11))
plt.title('normal')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.savefig('AP/patch_no_fgsm/normal_AP.jpg')
plt.show()

# ------------------------------------------mask------------------------------------------------------------------------
# 全為1
mask_percision = [1 for i in range(11)]
mask_label = []
for i in range(1, 6):
    mask_label.append('mask_{0}M(AP={1})'.format(
        i, round(sum(mask_percision)/11, 2)))

for i in range(len(mask_label)):
    plt.plot(recall, mask_percision, 's-', label=mask_label[i])

plt.xticks(np.linspace(0, 1, 11))
plt.yticks(np.linspace(0, 1, 11))
plt.title('mask')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.savefig('AP/patch_no_fgsm/mask_AP.jpg')
plt.show()'''

# ------------------------------------------normal_chin------------------------------------------------------------------------
# 全為1
normal_chin_percision = [1 for i in range(11)]
normal_chin_label = []
for i in range(1, 6):
    normal_chin_label.append('normal_chin_{0}M(AP={1})'.format(
        i, round(sum(normal_chin_percision)/11, 2)))

for i in range(len(normal_chin_label)):
    plt.plot(recall, normal_chin_percision, 's-', label=normal_chin_label[i])

plt.xticks(np.linspace(0, 1, 11))
plt.yticks(np.linspace(0, 1, 11))
plt.title('normal_chin')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.savefig('AP/patch_no_fgsm/normal_chin_AP.jpg')
plt.show()

# ------------------------------------------mask_chin------------------------------------------------------------------------

mask_chin_1M_percision = [1 for i in range(11)]
mask_chin_2M_percision = [1 for i in range(11)]
mask_chin_3M_percision = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
mask_chin_4M_percision = [1 for i in range(11)]
mask_chin_5M_percision = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]

mask_chin_percision = [mask_chin_1M_percision, mask_chin_2M_percision,
                       mask_chin_3M_percision, mask_chin_4M_percision, mask_chin_5M_percision]

mask_chin_label = []
for i in range(len(mask_chin_percision)):
    mask_chin_label.append('mask_chin_{0}M(AP={1})'.format(
        i+1, round(sum(mask_chin_percision[i])/11, 2)))

for i in range(len(mask_chin_label)):
    plt.plot(recall, mask_chin_percision[i], 's-', label=mask_chin_label[i])

plt.xticks(np.linspace(0, 1, 11))
plt.yticks(np.linspace(0, 1, 11))
plt.title('mask_chin')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.savefig('AP/patch_no_fgsm/mask_chin_AP.jpg')
plt.show()

# ------------------------------------------normal_forehead------------------------------------------------------------------------
# 全為1
normal_forehead_percision = [1 for i in range(11)]
normal_forehead_label = []
for i in range(1, 6):
    normal_forehead_label.append('normal_forehead_{0}M(AP={1})'.format(
        i, round(sum(normal_forehead_percision)/11, 2)))

for i in range(len(normal_forehead_label)):
    plt.plot(recall, normal_forehead_percision,
             's-', label=normal_forehead_label[i])

plt.xticks(np.linspace(0, 1, 11))
plt.yticks(np.linspace(0, 1, 11))
plt.title('normal_forehead')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.savefig('AP/patch_no_fgsm/normal_forehead_AP.jpg')
plt.show()

# ------------------------------------------mask_forehead------------------------------------------------------------------------

mask_forehead_1M_percision = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
mask_forehead_2M_percision = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
mask_forehead_3M_percision = [1 for i in range(11)]
mask_forehead_4M_percision = [1 for i in range(11)]
mask_forehead_5M_percision = [1, 1, 1, 1, 1, 1, 1, 1, 0.98, 0, 0]

mask_forehead_percision = [mask_forehead_1M_percision, mask_forehead_2M_percision,
                           mask_forehead_3M_percision, mask_forehead_4M_percision, mask_forehead_5M_percision]

mask_forehead_label = []
for i in range(len(mask_forehead_percision)):
    mask_forehead_label.append('mask_forehead_{0}M(AP={1})'.format(
        i+1, round(sum(mask_forehead_percision[i])/11, 2)))

for i in range(len(mask_forehead_label)):
    plt.plot(recall, mask_forehead_percision[i],
             's-', label=mask_forehead_label[i])

plt.xticks(np.linspace(0, 1, 11))
plt.yticks(np.linspace(0, 1, 11))
plt.title('mask_forehead')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.savefig('AP/patch_no_fgsm/mask_forehead_AP.jpg')
plt.show()

# ------------------------------------------normal_frame------------------------------------------------------------------------

normal_frame_percision = [1 for i in range(11)]

normal_frame_label = []
for i in range(1, 6):
    normal_frame_label.append('normal_frame_{0}M(AP={1})'.format(
        i, round(sum(normal_frame_percision)/11, 2)))

for i in range(len(normal_frame_label)):
    plt.plot(recall, normal_frame_percision, 's-', label=normal_frame_label[i])

plt.xticks(np.linspace(0, 1, 11))
plt.yticks(np.linspace(0, 1, 11))
plt.title('normal_frame')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.savefig('AP/patch_no_fgsm/normal_frame_AP.jpg')
plt.show()

# ------------------------------------------mask_frame------------------------------------------------------------------------

mask_frame_1M_percision = [1 for i in range(11)]
mask_frame_2M_percision = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
mask_frame_3M_percision = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0.96, 0]
mask_frame_4M_percision = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
mask_frame_5M_percision = [1, 1, 1, 1, 1, 1, 1, 1, 0.96, 0, 0]

mask_frame_percision = [mask_frame_1M_percision, mask_frame_2M_percision,
                        mask_frame_3M_percision, mask_frame_4M_percision, mask_frame_5M_percision]

mask_frame_label = []
for i in range(len(mask_frame_percision)):
    mask_frame_label.append('mask_frame_{0}M(AP={1})'.format(
        i+1, round(sum(mask_frame_percision[i])/11, 2)))

for i in range(len(mask_frame_label)):
    plt.plot(recall, mask_frame_percision[i], 's-', label=mask_frame_label[i])

plt.xticks(np.linspace(0, 1, 11))
plt.yticks(np.linspace(0, 1, 11))
plt.title('mask_frame')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.savefig('AP/patch_no_fgsm/mask_frame_AP.jpg')
plt.show()

# ------------------------------------------normal_mouth------------------------------------------------------------------------

normal_mouth_1M_percision = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
normal_mouth_2M_percision = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
normal_mouth_3M_percision = [1 for i in range(11)]
normal_mouth_4M_percision = [1 for i in range(11)]
normal_mouth_5M_percision = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]

normal_mouth_percision = [normal_mouth_1M_percision, normal_mouth_2M_percision,
                          normal_mouth_3M_percision, normal_mouth_4M_percision, normal_mouth_5M_percision]

normal_mouth_label = []
for i in range(len(normal_mouth_percision)):
    normal_mouth_label.append('normal_mouth_{0}M(AP={1})'.format(
        i+1, round(sum(normal_mouth_percision[i])/11, 2)))

for i in range(len(normal_mouth_label)):
    plt.plot(recall, normal_mouth_percision[i],
             's-', label=normal_mouth_label[i])

plt.xticks(np.linspace(0, 1, 11))
plt.yticks(np.linspace(0, 1, 11))
plt.title('normal_mouth')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.savefig('AP/patch_no_fgsm/normal_mouth_AP.jpg')
plt.show()

# ------------------------------------------mask_mouth------------------------------------------------------------------------

mask_mouth_1M_percision = [1 for i in range(11)]
mask_mouth_2M_percision = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
mask_mouth_3M_percision = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.96]
mask_mouth_4M_percision = [1, 1, 1, 1, 1, 1, 1, 1, 0.98, 0.98, 0]
mask_mouth_5M_percision = [1, 1, 1, 1, 1, 1, 0.97, 0.93, 0, 0, 0]

mask_mouth_percision = [mask_mouth_1M_percision, mask_mouth_2M_percision,
                        mask_mouth_3M_percision, mask_mouth_4M_percision, mask_mouth_5M_percision]

mask_mouth_label = []
for i in range(len(mask_mouth_percision)):
    mask_mouth_label.append('mask_mouth_{0}M(AP={1})'.format(
        i+1, round(sum(mask_mouth_percision[i])/11, 2)))

for i in range(len(mask_mouth_label)):
    plt.plot(recall, mask_mouth_percision[i], 's-', label=mask_mouth_label[i])

plt.xticks(np.linspace(0, 1, 11))
plt.yticks(np.linspace(0, 1, 11))
plt.title('mask_mouth')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.savefig('AP/patch_no_fgsm/mask_mouth_AP.jpg')
plt.show()
