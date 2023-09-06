import sys
import os
import time
sys.path.append(os.getcwd())

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation
import tqdm

# Bayesian optimization
# conda install -c conda-forge bayesian-optimization
from bayes_opt import BayesianOptimization, UtilityFunction

# load pyrenderer
from diffdvr import make_real3
from style.style_loss import StyleLoss, StyleLossType
from loaders.volume_loaders import load_dat_raw, load_nii
from loaders.tf_loaders import load_tf_from_xml
import pyrenderer

from vis import tfvis

# TF parameterization:
# color by Sigmoid, opacity by SoftPlus
class TransformTF(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.sigmoid = torch.nn.Sigmoid()
    self.softplus = torch.nn.Softplus()
  def forward(self, tf):
    assert len(tf.shape) == 3
    #assert tf.shape[2] == 5
    assert tf.shape[2] == 4
    return torch.cat([
      self.sigmoid(tf[:,:,0:3]), #color
      self.softplus(tf[:,:,3:4]), #opacity
      #tf[:,:,4:5] # position
      ], dim=2)

class InverseTransformTF(torch.nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, tf):
    def inverseSigmoid(y):
      return torch.log(-y/(y-1))
    def inverseSoftplus(y, beta=1, threshold=20):
      #if y*beta>threshold: return y
      return torch.log(torch.exp(beta*y)-1)/beta
    print(tf.shape)
    assert len(tf.shape) == 3
    #assert tf.shape[2] == 5
    assert tf.shape[2] == 4
    return torch.cat([
      inverseSigmoid(tf[:, :, 0:3]),  # color
      inverseSoftplus(tf[:, :, 3:4]),  # opacity
      #tf[:, :, 4:5]  # position
    ], dim=2)


def sample_cam_loss_function(yaw, pitch, distance):
  yaw = torch.tensor([[yaw]], dtype=dtype, device=device)
  pitch = torch.tensor([[pitch]], dtype=dtype, device=device)
  distance = torch.tensor([[distance]], dtype=dtype, device=device)
  viewport = pyrenderer.Camera.viewport_from_sphere(
    camera_center, yaw, pitch, distance, camera_orientation)
  ray_start, ray_dir = pyrenderer.Camera.generate_rays(
    viewport, fov_radians, W, H)
  inputs.camera = pyrenderer.CameraPerPixelRays(ray_start, ray_dir)
  pyrenderer.Renderer.render_forward(inputs, outputs)
  loss = torch.nn.functional.mse_loss(output_color, reference_color_gpu)
  return -loss.item()


# Bounded region of parameter space
pbounds = {
    'yaw': (np.radians(-180), np.radians(180)),
    'pitch': (np.radians(-90), np.radians(90)),
    'distance': (1.0, 1.0),
}


if __name__=='__main__':
  print(pyrenderer.__doc__)
  
  args_dict = {
    'l': 'W2'
  }
  
  # Parse the command line arguments.
  for i in range(1, len(sys.argv), 2):
    key = sys.argv[i][1:]
    value = sys.argv[i+1]
    # Guess the correct type
    if isinstance(args_dict[key], int):
      args_dict[key] = int(value)
    elif isinstance(args_dict[key], float):
      args_dict[key] = float(value)
    elif isinstance(args_dict[key], bool):
      args_dict[key] = bool(value)
    elif isinstance(args_dict[key], str):
      args_dict[key] = value
    elif args_dict[key] == None:
      args_dict[key] = value
    else:
      print('Error: Unhandled type.', file=sys.stderr)
      args_dict[key] = value

  loss_name = args_dict['l']
  loss_type = None
  if loss_name == 'W2':
    loss_type = StyleLossType.W2
  elif loss_name == 'Gram':
    loss_type = StyleLossType.GRAM
  elif loss_name == 'L2':
    loss_type = StyleLossType.L2
  else:
    raise Exception('Invalid loss type.')
  print(f'Optimizing for loss {loss_name}.')

  # Test case 0: 1D TF, Scalar/VisHuman Head [512 512 302] (CT)/vmhead.dat
  # Test case 1: 2D TF, Scalar/ImageVis3D/head.dat
  # Test case 2: 1D TF binary segmentation, OrganSegmentations/volume-2.nii
  tf_test_case_idx = 2
  optimize_camera = True
  optimize_tf = True
  camera_use_bayopt = True
  tf_forward_grad = False
  use_same_pos_opt = False
  
  # TODO: Testing
  if tf_test_case_idx != 0:
    optimize_camera = False
    use_same_pos_opt = True

  print("Create data set")
  # Data set test case
  data_test_case_idx = 2
  #volume = pyrenderer.Volume.create_implicit(pyrenderer.ImplicitEquation.CubeX, 64)
  #volume = pyrenderer.Volume.create_implicit(pyrenderer.ImplicitEquation.MarschnerLobb, 64)
  #data_array = load_dat_raw('/media/christoph/Elements/Datasets/Scalar/Head [512 512 106] (CT)/CT_Head_large.dat')
  if os.path.exists('/media/christoph/Elements/Datasets'):
    data_set_folder = '/media/christoph/Elements/Datasets'
  else:
    data_set_folder = '/mnt/data/Flow'
  # Scalar/VisHuman Head [256 256 256] (CT)/vmhead256cubed.dat
  if data_test_case_idx == 0:
    data_array = load_dat_raw(f'{data_set_folder}/Scalar/VisHuman Head [512 512 302] (CT)/vmhead.dat')
  elif data_test_case_idx == 1:
    data_array = load_dat_raw(f'{data_set_folder}/Scalar/ImageVis3D/head.dat')
  elif data_test_case_idx == 2:
    data_array = load_nii(f'{data_set_folder}/OrganSegmentations/volume-2.nii', normalize=True)
    label_array = np.round(load_nii(f'{data_set_folder}/OrganSegmentations/labels-2.nii', normalize=False))
  print(data_array.shape)
  volume = pyrenderer.Volume.from_numpy(data_array)
  volume.copy_to_gpu()
  print("density tensor: ", volume.getDataGpu(0).shape, volume.getDataGpu(0).dtype, volume.getDataGpu(0).device)

  B = 1 # batch dimension
  H = 256 # screen height
  W = 512 # screen width
  Y = volume.resolution.y
  Z = volume.resolution.z
  X = volume.resolution.x
  device = volume.getDataGpu(0).device
  dtype = volume.getDataGpu(0).dtype

  opacity_scaling = 50.0
  #tf_mode = pyrenderer.TFMode.Linear
  segmentation_mode = pyrenderer.SegmentationMode.SegmentationOff
  if tf_test_case_idx == 0:
    tf_mode = pyrenderer.TFMode.Texture
    res = 32
  elif tf_test_case_idx == 1:
    tf_mode = pyrenderer.TFMode.Texture2D
    res_x = 8
    res_y = 8
    res = res_x * res_y
  elif tf_test_case_idx == 2:
    tf_mode = pyrenderer.TFMode.Texture
    res_x = 32
    res = res_x * 2
    segmentation_mode = pyrenderer.SegmentationMode.SegmentationBinary
    optimize_segmentation_volume = False
    optimize_tf = True
  #tf = torch.tensor([[
  #  #r,g,b,a,pos
  #  [0.2313,0.2980,0.7529,0.0 *opacity_scaling,0],
  #  [0.5647,0.6980,0.9960,0.25*opacity_scaling,0.25],
  #  [0.8627,0.8627,0.8627,0.5 *opacity_scaling,0.5],
  #  [0.9607,0.6117,0.4901,0.75*opacity_scaling,0.75],
  #  [0.7058,0.0156,0.1490,1.0 *opacity_scaling,1]
  #]], dtype=dtype, device=device)
  #tf = torch.tensor([[
  #  #r,g,b,a,pos
  #  [0.9,0.01,0.01,0.001,0],
  #  [0.9,0.58,0.46,0.001,0.45],
  #  [0.9,0.61,0.50,0.8*opacity_scaling,0.5],
  #  [0.9,0.66,0.55,0.001,0.55],
  #  [0.9,0.99,0.99,0.001,1]
  #]], dtype=dtype, device=device)
  if tf_test_case_idx == 0:
    if data_test_case_idx == 0:
      tf_filename = f'{str(Path.home())}/Programming/C++/Correrender/Data/TransferFunctions/VisHumanHead2.xml'
    elif data_test_case_idx == 1:
      tf_filename = f'{str(Path.home())}/Programming/C++/Correrender/Data/TransferFunctions/VisHumanHeadB.xml'
    tf_array = load_tf_from_xml(tf_filename, res, opacity_scaling, write_pos=False)
    #tf_array = [[t[0], t[1], t[2], max(t[3], 1e-3), t[4]] for t in tf_array]
    tf_array = [[t[0], t[1], t[2], max(t[3], 1e-3)] for t in tf_array]
  elif tf_test_case_idx == 1:
    c0 = [1e-3, 1e-3, 1e-3, 1e-3]
    c1 = [0.0, 0.6, 1.0, opacity_scaling]
    tf_array = [c1 if 2 <= x <= 3 and 2 <= y <= 3 else c0 for x in range(res_x) for y in range(res_y)]
  elif tf_test_case_idx == 2:
    c0 = [1e-3, 1e-3, 1e-3, 1e-3]
    c1 = [1e-3, 0.6, 0.999, opacity_scaling * 0.1]
    c2 = [0.6, 0.999, 1e-3, opacity_scaling]
    #tf_array = \
    #  [c1 if 2 <= x <= 3 else c0 for x in range(res_x)] + \
    #  [c1 if 2 <= x <= 3 else c0 for x in range(res_x)]
    tf_array = \
      [c1 if 2 <= x <= 10 else c0 for x in range(res_x)] + \
      [c2 if 0 <= x <= 32 else c0 for x in range(res_x)]
  tf = torch.tensor([tf_array], dtype=dtype, device=device)

  print("Create renderer inputs")
  inputs = pyrenderer.RendererInputs()
  inputs.screen_size = pyrenderer.int2(W, H)
  inputs.volume = volume.getDataGpu(0)
  inputs.volume_filter_mode = pyrenderer.VolumeFilterMode.Trilinear
  inputs.box_min = pyrenderer.real3(-0.5, -0.5, -0.5)
  inputs.box_size = pyrenderer.real3(1, 1, 1)
  inputs.step_size = 0.5 / X
  inputs.tf_mode = tf_mode
  inputs.tf = tf
  inputs.blend_mode = pyrenderer.BlendMode.BeerLambert
  inputs.segmentation_mode = segmentation_mode
  if tf_test_case_idx == 0:
    #inputs.volume_grad = volume.getDataGpu(0)
    inputs.tf_res_x = res
  elif tf_test_case_idx == 1:
    volume_grad = pyrenderer.Volume.create_grad_volume_cpu(volume)
    volume_grad.copy_to_gpu()
    inputs.volume_grad = volume_grad.getDataGpu(0)
    inputs.tf_res_x = res_x
  elif tf_test_case_idx == 2:
    mask_val = 0
    mask_a = label_array == mask_val
    mask_b = label_array != mask_val
    label_array[mask_a] = -1e4
    label_array[mask_b] = 1e4
    #label_array = np.ones_like(label_array) * -1e4
    volume_segmentation = pyrenderer.Volume.from_numpy(label_array)
    volume_segmentation.copy_to_gpu()
    volume_segmentation_tensor_ref = volume_segmentation.getDataGpu(0)
    inputs.tf_res_x = res_x
    if optimize_segmentation_volume:
      volume_segmentation_tensor = torch.zeros_like(volume_segmentation_tensor_ref)
    else:
      volume_segmentation_tensor = volume_segmentation_tensor_ref.clone()
    inputs.volume_segmentation = volume_segmentation_tensor_ref

  # settings
  fov_degree = 45.0
  fov_radians = np.radians(fov_degree)
  use_yaw_pitch_ref = True
  if not use_yaw_pitch_ref:
    camera_origin = np.array([0.0, -0.71, -0.70])
    camera_lookat = np.array([0.0, 0.0, 0.0])
    camera_up = np.array([0,-1,0])
    invViewMatrix = pyrenderer.Camera.compute_matrix(
      make_real3(camera_origin), make_real3(camera_lookat), make_real3(camera_up),
      fov_degree, W, H)
    inputs.camera = invViewMatrix
    inputs.camera_mode = pyrenderer.CameraMode.InverseViewMatrix
    print("view matrix:")
    print(np.array(invViewMatrix))
  else:
    camera_ref_orientation = pyrenderer.Orientation.Ym
    camera_ref_center = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
    camera_ref_pitch = torch.tensor([[np.radians(-45)]], dtype=dtype, device=device)
    camera_ref_yaw = torch.tensor([[np.radians(-90)]], dtype=dtype, device=device)
    camera_ref_distance = torch.tensor([[1.0]], dtype=dtype, device=device)
    inputs.camera_mode = pyrenderer.CameraMode.RayStartDir
    viewport_ref = pyrenderer.Camera.viewport_from_sphere(
      camera_ref_center, camera_ref_yaw, camera_ref_pitch, camera_ref_distance, camera_ref_orientation)
    ray_start_ref, ray_dir_ref = pyrenderer.Camera.generate_rays(
      viewport_ref, fov_radians, W, H)
    inputs.camera = pyrenderer.CameraPerPixelRays(ray_start_ref, ray_dir_ref)

  print("Create forward difference settings")
  differences_settings_camera = pyrenderer.ForwardDifferencesSettings()
  #differences_settings.D = 4*5 # We want gradients for all control points
  #derivative_tf_indices = torch.tensor([[
  #  [0,1,2,3,-1],
  #  [4,5,6,7,-1],
  #  [8,9,10,11,-1],
  #  [12,13,14,15,-1],
  #  [16,17,18,19,-1]
  #]], dtype=torch.int32)
  #differences_settings.D = 4*3 # We want gradients for all inner control points
  #derivative_tf_indices = torch.tensor([[
  #  [-1,-1,-1,-1,-1],
  #  [0,1,2,3,-1],
  #  [4,5,6,7,-1],
  #  [8,9,10,11,-1],
  #  [-1, -1, -1, -1, -1]
  #]], dtype=torch.int32)
  #differences_settings.D = 4*res + 6 # We want gradients for all control points
  #tf_indices = [[i * 4 + j for j in range(4)] + [-1] for i in range(res)]
  #derivative_tf_indices = torch.tensor([tf_indices], dtype=torch.int32)
  #differences_settings.d_tf = derivative_tf_indices.to(device=device)
  #differences_settings.has_tf_derivatives = True
  #differences_settings.d_rayStart = pyrenderer.int3(4*res, 4*res + 1, 4*res + 2)
  #differences_settings.d_rayDir = pyrenderer.int3(4*res + 3, 4*res + 4, 4*res + 5)
  differences_settings_camera.D = 6
  differences_settings_camera.d_rayStart = pyrenderer.int3(0, 1, 2)
  differences_settings_camera.d_rayDir = pyrenderer.int3(3, 4, 5)

  if tf_forward_grad:
    differences_settings_tf = pyrenderer.ForwardDifferencesSettings()
    differences_settings_tf.D = 4*res # We want gradients for all control points
    tf_indices = [[i * 4 + j for j in range(4)] for i in range(res)]
    derivative_tf_indices = torch.tensor([tf_indices], dtype=torch.int32)
    differences_settings_tf.d_tf = derivative_tf_indices.to(device=device)
    differences_settings_tf.has_tf_derivatives = True
    gradients_out_tf = torch.empty(1, H, W, differences_settings_tf.D, 4, dtype=dtype, device=device)

  print("Create renderer outputs")
  output_color = torch.empty(1, H, W, 4, dtype=dtype, device=device)
  output_termination_index = torch.empty(1, H, W, dtype=torch.int32, device=device)
  outputs = pyrenderer.RendererOutputs(output_color, output_termination_index)
  gradients_out_cam = torch.empty(1, H, W, differences_settings_camera.D, 4, dtype=dtype, device=device)

  # render reference
  print("Render reference")
  pyrenderer.Renderer.render_forward(inputs, outputs)
  reference_color_gpu = output_color.clone()
  reference_color_image = output_color.cpu().numpy()[0]
  reference_tf = tf.cpu().numpy()[0]

  # set camera settings for optimization view
  if optimize_camera:
    camera_orientation = pyrenderer.Orientation.Ym
    camera_center = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
    camera_initial_pitch = torch.tensor([[np.radians(30)]], dtype=dtype, device=device) # torch.tensor([[np.radians(-14.5)]], dtype=dtype, device=device)
    camera_initial_yaw = torch.tensor([[np.radians(-20)]], dtype=dtype, device=device) # torch.tensor([[np.radians(113.5)]], dtype=dtype, device=device)
    camera_initial_distance = torch.tensor([[1.0]], dtype=dtype, device=device)
    inputs.camera_mode = pyrenderer.CameraMode.RayStartDir
    viewport = pyrenderer.Camera.viewport_from_sphere(
      camera_center, camera_initial_yaw, camera_initial_pitch, camera_initial_distance, camera_orientation)
    ray_start, ray_dir = pyrenderer.Camera.generate_rays(
      viewport, fov_radians, W, H)
    inputs.camera = pyrenderer.CameraPerPixelRays(ray_start, ray_dir)
    current_pitch = camera_initial_pitch.clone()
    current_yaw = camera_initial_yaw.clone()
    current_distance = camera_initial_distance.clone()
  else:
    if use_same_pos_opt:
      camera_origin = np.array([0.0, -0.71, -0.70])
      camera_lookat = np.array([0.0, 0.0, 0.0])
      camera_up = np.array([0,-1,0])
    else:
      camera_origin = np.array([0.0, -0.9, 0.44])
      camera_lookat = np.array([0.0, 0.0, 0.0])
      camera_up = np.array([0,0,1])
    invViewMatrix = pyrenderer.Camera.compute_matrix(
      make_real3(camera_origin), make_real3(camera_lookat), make_real3(camera_up),
      fov_degree, W, H)
    inputs.camera = invViewMatrix
    inputs.camera_mode = pyrenderer.CameraMode.InverseViewMatrix

  # initialize initial TF and render
  print("Render initial")
  #initial_tf = torch.tensor([[
  #  # r,g,b,a,pos
  #  [0.7058,0.0156,0.1490,1.0 *opacity_scaling,0],
  #  [0.9607,0.6117,0.4901,1.0 *opacity_scaling,0.25],
  #  [0.8627,0.8627,0.8627,1.0 *opacity_scaling,0.5],
  #  [0.5647,0.6980,0.9960,1.0 *opacity_scaling,0.75],
  #  [0.2313,0.2980,0.7529,1.0 *opacity_scaling,1]
  #]], dtype=dtype, device=device)
  #initial_tf = torch.tensor([[
  #  # r,g,b,a,pos
  #  [0.9,0.01,0.01,0.001,0],
  #  [0.2, 0.4, 0.3, 10, 0.45],
  #  [0.6, 0.7, 0.2, 7, 0.5],
  #  [0.5, 0.6, 0.4, 5, 0.55],
  #  [0.9,0.99,0.99,0.001,1]
  #]], dtype=dtype, device=device)
  #init_tf_array = [
  #  [
  #    0.5, 0.5, 0.5,
  #    max(i / (res - 1), 1e-3) * opacity_scaling,
  #    i / (res - 1)] \
  #  for i in range(res) ]
  if not optimize_tf:
    init_tf_array = tf_array
  elif tf_test_case_idx == 0:
    init_tf_array = [
      [0.5, 0.5, 0.5, max(i / (res - 1), 1e-3) * opacity_scaling] for i in range(res) ]
  elif tf_test_case_idx == 1:
    init_tf_array = [
      [0.5, 0.5, 0.5, 0.2 * opacity_scaling] for i in range(res) ]
  elif tf_test_case_idx == 2:
    init_tf_array = [
      [0.5, 0.5, 0.5, 0.2 * opacity_scaling] for i in range(res) ]
  initial_tf = torch.tensor([init_tf_array], dtype=dtype, device=device)
  print("Initial tf (original):", initial_tf)
  inputs.tf = initial_tf
  inputs.volume_segmentation = volume_segmentation_tensor
  pyrenderer.Renderer.render_forward(inputs, outputs)
  initial_color_image = output_color.cpu().numpy()[0]
  tf = InverseTransformTF()(initial_tf)
  print("Initial tf (transformed):", tf)
  initial_tf = initial_tf.cpu().numpy()[0]
      
  if not tf_forward_grad:
    adjoint_outputs = pyrenderer.AdjointOutputs()
    if optimize_tf:
      adjoint_outputs.has_tf_derivatives = True
      adjoint_outputs.tf_delayed_accumulation = True
      grad_tf = torch.zeros_like(tf)
      adjoint_outputs.adj_tf = grad_tf
    if optimize_segmentation_volume:
      grad_segmentation_volume = torch.zeros_like(volume_segmentation_tensor)
      adjoint_outputs.has_segmentation_volume_derivatives = True
      adjoint_outputs.adj_segmentation_volume = grad_segmentation_volume

  # Construct the model
  class RendererDerivCamera(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ray_start, ray_dir):
      inputs.camera = pyrenderer.CameraPerPixelRays(ray_start, ray_dir)
      # render
      pyrenderer.Renderer.render_forward_gradients(inputs, differences_settings_camera, outputs, gradients_out_cam)
      ctx.save_for_backward(gradients_out_cam)
      return output_color
    @staticmethod
    def backward(ctx, grad_output_color):
      gradients_out_cam, = ctx.saved_tensors
      # apply forward derivatives to the adjoint of the color
      # to get the adjoint of the tf
      grad_output_color = grad_output_color.unsqueeze(3) # for broadcasting over the derivatives
      gradients = torch.mul(gradients_out_cam, grad_output_color) # adjoint-multiplication
      gradients = torch.sum(gradients, dim=4)  # reduce over channel
      grad_ray_start = gradients[..., 0:3]
      grad_ray_dir = gradients[..., 3:6]
      return grad_ray_start, grad_ray_dir
  rendererDerivCamera = RendererDerivCamera.apply

  class OptimModelCamera(torch.nn.Module):
    def __init__(self):
      super().__init__()
    def forward(self, transformed_tf, current_pitch, current_yaw, current_distance):
      viewport = pyrenderer.Camera.viewport_from_sphere(
        camera_center, current_yaw, current_pitch, current_distance, camera_orientation)
      ray_start, ray_dir = pyrenderer.Camera.generate_rays(
        viewport, fov_radians, W, H)
      inputs.tf = transformed_tf
      color = rendererDerivCamera(ray_start, ray_dir)
      loss = torch.nn.functional.mse_loss(color, reference_color_gpu)
      return loss, viewport, color
      
  # Construct the model
  class RendererDerivTFForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, transformed_tf):
      inputs.tf = transformed_tf
      # render
      pyrenderer.Renderer.render_forward_gradients(inputs, differences_settings_tf, outputs, gradients_out_tf)
      ctx.save_for_backward(transformed_tf, gradients_out_tf)
      return output_color
    @staticmethod
    def backward(ctx, grad_output_color):
      current_tf, gradients_out_tf = ctx.saved_tensors
      # apply forward derivatives to the adjoint of the color
      # to get the adjoint of the tf
      grad_output_color = grad_output_color.unsqueeze(3) # for broadcasting over the derivatives
      gradients = torch.mul(gradients_out_tf, grad_output_color) # adjoint-multiplication
      gradients = torch.sum(gradients, dim=[1,2,4]) # reduce over screen height, width and channel
      # map to output variables
      grad_tf = torch.zeros_like(current_tf)
      for R in range(grad_tf.shape[1]):
        for C in range(grad_tf.shape[2]):
          idx = derivative_tf_indices[0,R,C]
          if idx>=0:
            grad_tf[:,R,C] = gradients[:,idx]
      return grad_tf
      
  # Construct the model
  class RendererDerivTFAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, transformed_tf):
      inputs.tf = transformed_tf
      # render
      grad_tf.zero_()
      pyrenderer.Renderer.render_forward(inputs, outputs)
      return output_color
    @staticmethod
    def backward(ctx, grad_output_color):
      pyrenderer.Renderer.render_adjoint(inputs, outputs, grad_output_color, adjoint_outputs)
      return grad_tf
  
  if tf_forward_grad:
    rendererDerivTF = RendererDerivTFForward.apply
  else:
    rendererDerivTF = RendererDerivTFAdjoint.apply

  class OptimModelTF(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.style_loss = StyleLoss(reference_color_gpu.permute(0, 3, 1, 2)[:, 0:3, :, :], loss_type=loss_type)
    def forward(self, transformed_tf):
      color = rendererDerivTF(transformed_tf)
      #loss = torch.nn.functional.mse_loss(color, reference_color_gpu)
      loss = self.style_loss(color.permute(0, 3, 1, 2)[:, 0:3, :, :])
      return loss, transformed_tf, color
      
  # Construct the model
  class RendererDerivTFSegmentationAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, transformed_tf, volume_segmentation_tensor):
      inputs.tf = transformed_tf
      inputs.volume_segmentation = volume_segmentation_tensor
      # render
      grad_tf.zero_()
      grad_segmentation_volume.zero_()
      pyrenderer.Renderer.render_forward(inputs, outputs)
      return output_color
    @staticmethod
    def backward(ctx, grad_output_color):
      pyrenderer.Renderer.render_adjoint(inputs, outputs, grad_output_color, adjoint_outputs)
      return grad_tf, grad_segmentation_volume

  rendererDerivTFSegmentation = RendererDerivTFSegmentationAdjoint.apply

  class OptimModelTFSegmentation(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.style_loss = StyleLoss(reference_color_gpu.permute(0, 3, 1, 2)[:, 0:3, :, :], loss_type=loss_type)
    def forward(self, transformed_tf, volume_segmentation_tensor):
      color = rendererDerivTFSegmentation(transformed_tf, volume_segmentation_tensor)
      #loss = torch.nn.functional.mse_loss(color, reference_color_gpu)
      loss = self.style_loss(color.permute(0, 3, 1, 2)[:, 0:3, :, :])
      return loss, transformed_tf, volume_segmentation_tensor, color

  # Construct the model
  class RendererDerivSegmentationAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume_segmentation_tensor):
      inputs.volume_segmentation = volume_segmentation_tensor
      # render
      grad_segmentation_volume.zero_()
      pyrenderer.Renderer.render_forward(inputs, outputs)
      return output_color
    @staticmethod
    def backward(ctx, grad_output_color):
      pyrenderer.Renderer.render_adjoint(inputs, outputs, grad_output_color, adjoint_outputs)
      return grad_segmentation_volume

  rendererDerivSegmentation = RendererDerivSegmentationAdjoint.apply

  class OptimModelSegmentation(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.style_loss = StyleLoss(reference_color_gpu.permute(0, 3, 1, 2)[:, 0:3, :, :], loss_type=loss_type)
    def forward(self, volume_segmentation_tensor):
      color = rendererDerivSegmentation(volume_segmentation_tensor)
      #loss = torch.nn.functional.mse_loss(color, reference_color_gpu)
      loss = self.style_loss(color.permute(0, 3, 1, 2)[:, 0:3, :, :])
      return loss, volume_segmentation_tensor, color

  model_camera = OptimModelCamera()
  if optimize_tf:
    if optimize_segmentation_volume:
      model_tf = OptimModelTFSegmentation()
    else:
      model_tf = OptimModelTF()
  elif optimize_segmentation_volume:
    model_tf = OptimModelSegmentation()

  start_time = time.time()

  # run optimization
  tf_transform = TransformTF()
  iterations = 400  # 400
  reconstructed_color = []
  reconstructed_tf = []
  reconstructed_viewport = []
  reconstructed_loss = []
  current_tf = tf.clone()
  if optimize_tf:
    current_tf.requires_grad_()
  if optimize_segmentation_volume:
    volume_segmentation_tensor.requires_grad_()
  variables = []
  optimize_pitch = True
  optimize_yaw = True
  optimize_distance = False
  if optimize_camera:
    if optimize_pitch:
      current_pitch.requires_grad_()
      variables.append(current_pitch)
    if optimize_yaw:
      current_yaw.requires_grad_()
      variables.append(current_yaw)
    if optimize_distance:
      current_distance.requires_grad_()
      variables.append(current_distance)
    optimizer_camera = torch.optim.Adam(variables, lr=0.2)
    if camera_use_bayopt:
      last_cam_loss = np.inf
      # Default is UCB with kappa=2.576
      # For more details see: https://github.com/bayesian-optimization/BayesianOptimization/blob/master/examples/exploitation_vs_exploration.ipynb
      acquisition_function = UtilityFunction(kind="ucb", kappa=10)
      random_state = np.random.RandomState(17)
  tf_optim_variables = []
  if optimize_tf:
    tf_optim_variables.append(current_tf)
  if optimize_segmentation_volume:
    tf_optim_variables.append(volume_segmentation_tensor)
  optimizer_tf = torch.optim.Adam(tf_optim_variables, lr=0.2)
  #optimizer = torch.optim.LBFGS(variables, lr=0.2)
  for iteration in range(iterations):
    # TODO: softplus for opacity, sigmoid for color
    transformed_tf = tf_transform(current_tf)
    if optimize_camera:
      if camera_use_bayopt and iteration % 20 == 0:
        bayesian_optimizer = BayesianOptimization(
            f=sample_cam_loss_function,
            pbounds=pbounds,
            random_state=random_state,
        )
        #bayesian_optimizer.probe(
        #    params={'yaw': np.radians(-90), 'pitch': np.radians(-45), 'distance': 1.0},
        #    lazy=True,
        #)
        bayesian_optimizer.maximize(
            init_points=10,
            n_iter=30,
            acquisition_function=acquisition_function
        )
        optimal_loss = -bayesian_optimizer.max['target']
        if optimal_loss < last_cam_loss:
          print(f'Updated camera loss {last_cam_loss:7.5f} with BOS loss {optimal_loss:7.5f}')
          last_cam_loss = optimal_loss
          with torch.no_grad():
            current_yaw.copy_(torch.tensor([[bayesian_optimizer.max['params']['yaw']]], dtype=dtype))
            current_pitch.copy_(torch.tensor([[bayesian_optimizer.max['params']['pitch']]], dtype=dtype))
            current_distance.copy_(torch.tensor([[bayesian_optimizer.max['params']['distance']]], dtype=dtype))
        else:
          print(f'Did NOT update camera loss {last_cam_loss:7.5f} with BOS loss {optimal_loss:7.5f}')
      optimizer_camera.zero_grad()
      loss_camera, current_viewport, color = model_camera(transformed_tf, current_pitch, current_yaw, current_distance)
      loss_camera.backward()
      optimizer_camera.step()
      current_viewport = pyrenderer.Camera.viewport_from_sphere(
        camera_center, current_yaw, current_pitch, current_distance, camera_orientation)
      reconstructed_viewport.append(current_viewport.detach().cpu().numpy()[0])
      viewport = pyrenderer.Camera.viewport_from_sphere(
        camera_center, current_yaw, current_pitch, current_distance, camera_orientation)
      ray_start, ray_dir = pyrenderer.Camera.generate_rays(
        viewport, fov_radians, W, H)
      inputs.camera = pyrenderer.CameraPerPixelRays(ray_start, ray_dir)
      if camera_use_bayopt:
        last_cam_loss = loss_camera.item()
    optimizer_tf.zero_grad()
    if optimize_tf:
      if optimize_segmentation_volume:
        loss_tf, transformed_tf, volume_segmentation_tensor, color = model_tf(transformed_tf, volume_segmentation_tensor)
      else:
        loss_tf, transformed_tf, color = model_tf(transformed_tf)
    else:
      loss_tf, volume_segmentation_tensor, color = model_tf(volume_segmentation_tensor)
    #if optimize_segmentation_volume:
    #  grad_test = grad_segmentation_volume.detach().cpu().numpy()
    #  print(f'sum: {np.sum(grad_test)}')
    reconstructed_color.append(color.detach().cpu().numpy()[0,:,:,0:3])
    reconstructed_loss.append(loss_tf.item())
    reconstructed_tf.append(transformed_tf.detach().cpu().numpy()[0])
    loss_tf.backward()
    optimizer_tf.step()
    if optimize_camera:
      print("Iteration % 4d, Loss_Cam %7.5f Loss_TF %7.5f"%(iteration, loss_camera.item(), loss_tf.item()))
    else:
      print("Iteration % 4d, Loss %7.5f"%(iteration, loss_tf.item()))

  elapsed_time = time.time() - start_time
  print(f'Elapsed time optimization: {elapsed_time}s')

  print("Visualize Optimization")
  fig, axs = plt.subplots(3, 2, figsize=(8,6))
  axs[0,0].imshow(reference_color_image[:,:,0:3])
  if tf_test_case_idx != 1:
    tfvis.renderTfTexture(reference_tf, axs[0, 1])
  else:
    tfvis.renderTfTexture2d(reference_tf, res_x, axs[0, 1])
  axs[1, 0].imshow(reconstructed_color[0])
  if tf_test_case_idx != 1:
    tfvis.renderTfTexture(reconstructed_tf[0], axs[1, 1])
  else:
    tfvis.renderTfTexture2d(reconstructed_tf[0], res_x, axs[1, 1])
  axs[2,0].imshow(initial_color_image[:,:,0:3])
  if tf_test_case_idx != 1:
    tfvis.renderTfTexture(initial_tf, axs[2, 1])
  else:
    tfvis.renderTfTexture2d(initial_tf, res_x, axs[2, 1])
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

  print("Write frames")
  with tqdm.tqdm(total=len(reconstructed_color)) as pbar:
    def update(frame):
      axs[1, 0].imshow(reconstructed_color[frame])
      if tf_test_case_idx != 1:
        tfvis.renderTfTexture(reconstructed_tf[frame], axs[1, 1])
      else:
        tfvis.renderTfTexture2d(reconstructed_tf[frame], res_x, axs[1, 1])
      fig.suptitle("Iteration % 4d, Loss: %7.5f"%(frame, reconstructed_loss[frame]))
      if frame>0: pbar.update(1)
    # , cache_frame_data=False
    anim = matplotlib.animation.FuncAnimation(fig, update, frames=len(reconstructed_color), blit=False)
    anim.save(f"test_tf_optimization_{loss_name}.mp4")
  #plt.show()

  pyrenderer.cleanup()
