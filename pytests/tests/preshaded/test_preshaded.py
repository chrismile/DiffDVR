import sys
import os
import time
import array
sys.path.append(os.getcwd())

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation
import tqdm
import json

from PIL import Image

#import h5py
from netCDF4 import Dataset

#conda install -c conda-forge openexr-python
import OpenEXR
import Imath

# load pyrenderer
from diffdvr import make_real3
from style.style_loss import StyleLoss, StyleLossType
from loaders.volume_loaders import load_dat_raw, load_nii
import pyrenderer

from vis import tfvis


if __name__=='__main__':
  print(pyrenderer.__doc__)

  #test_case = 'wholebody'
  test_case = 'cloud'
  #test_case = 'head'

  if test_case == 'wholebody':
    cameras_dir = '/home/neuhauser/Programming/DL/vpt_denoise/out_2024-03-30_13:58:13/'
    #cameras_dir = '/home/neuhauser/Programming/DL/vpt_denoise/out_2024-03-30_13:58:13_test/'
    #X = 512
    #Y = 512
    #Z = 3172
    X = 256
    Y = 256
    Z = 1586
    sx = 0.5 * 512.0 / 3172.0
    sy = 0.5 * 512.0 / 3172.0
    sz = 0.5
  elif test_case == 'cloud':
    cameras_dir = '/home/neuhauser/datasets/VPT/scatter_cloud_full_2048_exr/images/'
    #cameras_dir = '/home/neuhauser/Programming/DL/vpt_denoise/out_2024-06-25_16:49:37/'
    #cameras_dir = '/home/neuhauser/Programming/DL/vpt_denoise/out_2024-06-27_20:17:57/'
    X = 1987 // 4
    Y = 1351 // 4
    Z = 2449 // 4
    sx = 1987.0 / 2449.0
    sy = 1351.0 / 2449.0
    sz = 2449.0 / 2449.0
    #sx = 0.5 * 1987.0 / 2449.0
    #sy = 0.5 * 1351.0 / 2449.0
    #sz = 0.5 * 2449.0 / 2449.0
  elif test_case == 'head':
    cameras_dir = '/home/neuhauser/Programming/DL/vpt_denoise/out_2024-06-28_12:48:49/'
    #cameras_dir = '/home/neuhauser/Programming/DL/vpt_denoise/out_2024-06-28_12:48:49_test/'
    X = 256
    Y = 256
    Z = 256
    sx = 0.5 * 0.2 / 0.2
    sy = 0.5 * 0.2 / 0.2
    sz = 0.5 * 0.15 / 0.2

  cameras_path = cameras_dir + 'cameras.json'
  with open(cameras_path) as f:
    cameras_json = json.load(f)
    camera0_json = cameras_json[0]
    if 'aabb' in camera0_json:
      aabb_json = camera0_json['aabb']
      sx = aabb_json[1] - aabb_json[0]
      sy = aabb_json[3] - aabb_json[2]
      sz = aabb_json[5] - aabb_json[4]
  
  device = "cuda"
  dtype = torch.float32

  B = 1 # batch dimension
  H = 1024 # screen height
  W = 1024 # screen width
  #device = volume.getDataGpu(0).device
  #dtype = volume.getDataGpu(0).dtype

  test_second_view = False
  opacity_scaling = 1.0
  tf = torch.tensor([[
    [0.0,0.0,0.0,0.0 *opacity_scaling],
    [1.0,1.0,1.0,1.0 *opacity_scaling]
  ]], dtype=dtype, device=device)

  print("Create data set")
  volume_tensor = torch.ones((4, X, Y, Z), dtype=dtype, device=device) * 0.5
  volume_tensor[3,:,:,:] = opacity_scaling
  #volume_densities = VolumePreshaded()
  #reference_volume_data = pyrenderer.TFUtils.preshade_volume(
  #    reference_volume_data, tf_reference, rs.tf_mode)

  print("Create renderer inputs")
  inputs = pyrenderer.RendererInputs()
  inputs.screen_size = pyrenderer.int2(W, H)
  inputs.volume = volume_tensor
  inputs.volume_filter_mode = pyrenderer.VolumeFilterMode.Preshaded
  inputs.box_min = pyrenderer.real3(-0.5 * sx, -0.5 * sy, -0.5 * sz)
  inputs.box_size = pyrenderer.real3(sx, sy, sz)
  inputs.step_size = 0.25 / max(X, max(Y, Z))
  inputs.tf_mode = pyrenderer.TFMode.Preshaded
  inputs.tf = tf
  inputs.blend_mode = pyrenderer.BlendMode.BeerLambert

  output_color_test = torch.empty(1, H, W, 4, dtype=dtype, device=device)
  output_termination_index_test = torch.empty(1, H, W, dtype=torch.int32, device=device)
  outputs_test = pyrenderer.RendererOutputs(output_color_test, output_termination_index_test)

  print("Create renderer outputs")
  output_color = torch.empty(1, H, W, 4, dtype=dtype, device=device)
  output_termination_index = torch.empty(1, H, W, dtype=torch.int32, device=device)
  outputs = pyrenderer.RendererOutputs(output_color, output_termination_index)

  adjoint_outputs = pyrenderer.AdjointOutputs()
  grad_volume = torch.zeros_like(volume_tensor)
  adjoint_outputs.has_volume_derivatives = True
  adjoint_outputs.adj_volume = grad_volume

  camera_ref_orientation = pyrenderer.Orientation.Ym
  camera_ref_center = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
  camera_ref_distance = torch.tensor([[0.7]], dtype=dtype, device=device)
  camera_test_pitch = torch.tensor([[np.radians(-45)]], dtype=dtype, device=device)
  camera_test_yaw = torch.tensor([[np.radians(-40)]], dtype=dtype, device=device)
  viewport_test = pyrenderer.Camera.viewport_from_sphere(
    camera_ref_center, camera_test_yaw, camera_test_pitch, camera_ref_distance, camera_ref_orientation)
  ray_start_test, ray_dir_test = pyrenderer.Camera.generate_rays(viewport_test, np.radians(45.0), W, H)
  camera_test_second_view = pyrenderer.CameraPerPixelRays(ray_start_test, ray_dir_test)

  output_color_test = torch.empty(1, H, W, 4, dtype=dtype, device=device)
  output_termination_index_test = torch.empty(1, H, W, dtype=torch.int32, device=device)
  outputs_test = pyrenderer.RendererOutputs(output_color_test, output_termination_index_test)

  # Construct the model
  class RendererDerivAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume_tensor):
      inputs.volume = volume_tensor
      # render
      grad_volume.zero_()
      pyrenderer.Renderer.render_forward(inputs, outputs)
      return output_color
    @staticmethod
    def backward(ctx, grad_output_color):
      pyrenderer.Renderer.render_adjoint(inputs, outputs, grad_output_color, adjoint_outputs)
      return grad_volume

  rendererDeriv = RendererDerivAdjoint.apply

  class OptimModelVolume(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.sigmoid = torch.nn.Sigmoid()
    def forward(self, iteration, volume_tensor):
      color = rendererDeriv(volume_tensor)
      loss = torch.nn.functional.mse_loss(color, reference_color_gpu)
      return loss, volume_tensor, color

  model = OptimModelVolume()

  start_time = time.time()

  # run optimization
  write_video = False
  write_hdf5 = False
  write_nc = True
  #epochs = 1
  epochs = 128
  iterations = len(cameras_json) * epochs
  reconstructed_color = []
  reconstructed_loss = []
  second_view_images = []
  reference_color_images = []
  volume_tensor.requires_grad_()
  variables = []
  variables.append(volume_tensor)
  #learning_rate = 0.05
  learning_rate = 0.02
  optimizer = torch.optim.Adam(variables, lr=learning_rate)
  #optimizer = torch.optim.LBFGS(variables, lr=0.2)
  for iteration in range(iterations):
    # TODO: softplus for opacity, sigmoid for color
    optimizer.zero_grad()
    
    img_idx = iteration % len(cameras_json)
    camera_json = cameras_json[img_idx]
    if 'fg_name' in camera_json:
      fg_image_path = cameras_dir + camera_json["fg_name"]
    else:
      fg_image_path = cameras_dir + f'img_{img_idx}.exr'

    if fg_image_path.endswith('.exr'):
      pt = Imath.PixelType(Imath.PixelType.FLOAT)
      ref_exr = OpenEXR.InputFile(fg_image_path)
      reference_color_image = np.array([array.array('f', ref_exr.channel(ch, pt)).tolist() for ch in ("R", "G", "B", "A")], dtype=np.float32)
      dw = ref_exr.header()["dataWindow"]
      reference_color_image = reference_color_image.reshape((1, 4, dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1))
      reference_color_image = reference_color_image.transpose(0, 2, 3, 1)
    elif fg_image_path.endswith('.png'):
      ref_img = Image.open(fg_image_path)
      reference_color_image = np.array(ref_img).astype(np.float32) / 255.0
      reference_color_image = reference_color_image.reshape((1, reference_color_image.shape[0], reference_color_image.shape[1], 4))
      reference_color_image = reference_color_image.transpose(0, 3, 1, 2)

    #reference_color_image = output_color.cpu().numpy()[0]
    if write_video:
      reference_color_images.append(reference_color_image[0,:,:,0:3])
    reference_color_gpu = torch.tensor(reference_color_image, device=device)

    if camera_json["fovy"] > 4.0:
      fovy = camera_json["fovy"]
    else:
      fovy = np.degrees(camera_json["fovy"])
    camera_origin = np.array(camera_json["position"])
    camera_origin[2] = -camera_origin[2]
    camera_right = np.array(camera_json["rotation"][0])
    camera_right[2] = -camera_right[2]
    camera_up = np.array(camera_json["rotation"][1])
    #camera_up[2] = -camera_up[2]
    camera_up[0] = -camera_up[0]
    camera_up[1] = -camera_up[1]
    camera_front = np.array(camera_json["rotation"][2])
    camera_front[0] = -camera_front[0]
    camera_front[1] = -camera_front[1]
    invViewMatrix = pyrenderer.Camera.compute_matrix2(
      make_real3(camera_origin), make_real3(camera_right), make_real3(camera_up), make_real3(camera_front), fovy, W, H)
    inputs.camera = invViewMatrix
    inputs.camera_mode = pyrenderer.CameraMode.InverseViewMatrix

    loss, volume_tensor, color = model(iteration, volume_tensor)
    if write_video:
      reconstructed_color.append(color.detach().cpu().numpy()[0,:,:,0:3])
    reconstructed_loss.append(loss.item())
    loss.backward()
    #volume_tensor_grad = volume_tensor.grad
    #print(volume_tensor_grad)
    optimizer.step()
    with torch.no_grad():
      inputs_cpy = inputs.clone()
      inputs_cpy.camera_mode = pyrenderer.CameraMode.RayStartDir
      inputs_cpy.camera = camera_test_second_view
      pyrenderer.Renderer.render_forward(inputs_cpy, outputs_test)
      test_image = output_color_test.cpu().numpy()[0]
      if write_video:
        second_view_images.append(test_image)
    print("Iteration % 4d, Loss %7.5f"%(iteration, loss.item()))

  elapsed_time = time.time() - start_time
  print(f'Elapsed time optimization: {elapsed_time}s')

  #if write_hdf5:
  #  with h5py.File('test_preshaded.hdf5', 'w') as hdf5_file:
  #    volumes = hdf5_file.create_dataset(
  #        "volume", (4, Z, Y, X), dtype=np.float32, chunks=(4, Z, Y, X))
  #    volumes[:, :, :, :] = volume_tensor.detach().cpu().numpy()

  if write_nc:
    ncfile = Dataset('test_preshaded.nc', mode='w', format='NETCDF4_CLASSIC')
    cdim = ncfile.createDimension('c', 4)
    zdim = ncfile.createDimension('z', Z)
    ydim = ncfile.createDimension('y', Y)
    xdim = ncfile.createDimension('x', X)
    outfield_color = ncfile.createVariable('color', np.float32, ('c', 'z', 'y', 'x'))
    #outfield_color[:, :, :, :] = volume_tensor.detach().cpu().numpy().flatten('F')
    #outfield_color.flatten('F') = volume_tensor.detach().cpu().numpy().flatten('F')
    outfield_color[:, :, :, :] = np.flip(volume_tensor.detach().cpu().numpy().transpose(0, 3, 2, 1), 1)
    ncfile.close()

  if write_video:
    print("Visualize Optimization")
    fig, axs = plt.subplots(3, 1, figsize=(8,6))
    axs[0].imshow(reference_color_images[0][:,:,0:3])
    axs[1].imshow(reconstructed_color[0])
    axs[2].imshow(second_view_images[0][:,:,0:3])
    axs[0].set_title("Color")
    axs[0].set_ylabel("Reference")
    axs[1].set_ylabel("Optimization")
    axs[2].set_ylabel("Side View")
    for i in range(3):
      axs[i].set_xticks([])
      #if j==0: axs[i,j].set_yticks([])
      axs[i].set_yticks([])
    fig.suptitle("Iteration % 4d, Loss: %7.3f" % (0, reconstructed_loss[0]))
    fig.tight_layout()

    print("Write frames")
    with tqdm.tqdm(total=len(reconstructed_color)) as pbar:
      def update(frame):
        axs[0].clear()
        axs[0].imshow(reference_color_images[frame][:,:,0:3])
        axs[1].clear()
        axs[1].imshow(reconstructed_color[frame])
        axs[2].clear()
        axs[2].imshow(second_view_images[frame][:,:,0:3])
        fig.suptitle("Iteration % 4d, Loss: %7.5f"%(frame, reconstructed_loss[frame]))
        if frame > 0:
          pbar.update(1)
      anim = matplotlib.animation.FuncAnimation(
        fig, update, frames=len(reconstructed_color), cache_frame_data=False)
      anim.save(f"test_preshaded.mp4")

  pyrenderer.cleanup()
