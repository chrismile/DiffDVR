import sys
import os
import time

import numpy

sys.path.append(os.getcwd())

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation
import tqdm

from loaders.tf_loaders import load_tf_from_xml
from vis import tfvis

opacity_scaling = 50.0
res = 32
data_test_case_idx = 0
if data_test_case_idx == 0:
    tf_filename = f'{str(Path.home())}/Programming/C++/Correrender/Data/TransferFunctions/VisHumanHead2.xml'
elif data_test_case_idx == 1:
    tf_filename = f'{str(Path.home())}/Programming/C++/Correrender/Data/TransferFunctions/VisHumanHeadB.xml'
tf_array = load_tf_from_xml(tf_filename, res, opacity_scaling, write_pos=False)
tf_array = np.array(tf_array)

num_its = 10
reconstructed_color = []
reconstructed_tf = []
reconstructed_loss = []
for i in range(num_its):
    reconstructed_color.append(numpy.ones((512, 256, 4)))
    reconstructed_tf.append(tf_array)
    reconstructed_loss.append(0.1)
reference_color_image = numpy.ones((512, 256, 4))
initial_color_image = numpy.ones((512, 256, 4))
reference_tf = tf_array
initial_tf = tf_array

fig, axs = plt.subplots(3, 2, figsize=(8,6))
axs[0,0].imshow(reference_color_image[:,:,0:3])
tfvis.renderTfTexture(reference_tf, axs[0, 1])
axs[1, 0].imshow(reconstructed_color[0])
tfvis.renderTfTexture(reconstructed_tf[0], axs[1, 1])
axs[2,0].imshow(initial_color_image[:,:,0:3])
tfvis.renderTfTexture(initial_tf, axs[2, 1])
axs[0,0].set_title("Color")
axs[0,1].set_title("Transfer Function")
axs[0,0].set_ylabel("Reference")
axs[1,0].set_ylabel("Optimization")
axs[2,0].set_ylabel("Initial")
for i in range(3):
  for j in range(2):
    axs[i,j].set_xticks([])
    if j==0: axs[i,j].set_yticks([])
fig.suptitle("Iteration % 4d, Loss: %7.3f" % (0, reconstructed_loss[0]))
fig.tight_layout()

current_time = time.time()
time_array = []
with tqdm.tqdm(total=len(reconstructed_color)) as pbar:
    def update(frame):
        global current_time
        #fig.clear(keep_observers=True)

        # TODO
        #axs[0, 0].imshow(reference_color_image[:, :, 0:3])
        #tfvis.renderTfTexture(reference_tf, axs[0, 1])
        #axs[1, 0].imshow(reconstructed_color[0])
        #tfvis.renderTfTexture(reconstructed_tf[0], axs[1, 1])
        #axs[2, 0].imshow(initial_color_image[:, :, 0:3])
        #tfvis.renderTfTexture(initial_tf, axs[2, 1])
        #axs[0, 0].set_title("Color")
        #axs[0, 1].set_title("Transfer Function")
        #axs[0, 0].set_ylabel("Reference")
        #axs[1, 0].set_ylabel("Optimization")
        #axs[2, 0].set_ylabel("Initial")
        #for i in range(3):
        #    for j in range(2):
        #        axs[i, j].set_xticks([])
        #        if j == 0: axs[i, j].set_yticks([])

        axs[1, 0].clear()
        axs[1, 0].imshow(reconstructed_color[frame])
        axs[1, 1].clear()
        #[p.remove() for p in axs[1, 1].patches]
        tfvis.renderTfTexture(reconstructed_tf[frame], axs[1, 1])
        fig.suptitle("Iteration % 4d, Loss: %7.5f" % (frame, reconstructed_loss[frame]))
        if frame > 0:
            pbar.update(1)
        next_time = time.time()
        time_array.append(next_time - current_time)
        current_time = next_time

    # , cache_frame_data=False
    anim = matplotlib.animation.FuncAnimation(
        fig, update, frames=len(reconstructed_color),
        cache_frame_data=False)
    anim.save(f"test_video.mp4")

plt.figure()
plt.plot(time_array)
plt.show()
