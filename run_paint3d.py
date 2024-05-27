import os
import cv2

image_dir = 'data/MOW/data/images'
mask_dir = 'data/MOW/data/masks/object'
masked_obj_image_dir = 'data/MOW/data/masked_obj_images'
watertight_mesh_dir = 'data/MOW/data/watertight_models' # TODO: Change to 'watertight_models_fine'
stage1_output_dir = 'outputs/stage1'
stage2_output_dir = 'outputs/stage2'

if not os.path.exists(masked_obj_image_dir):
    os.makedirs(masked_obj_image_dir)


watertight_meshes = os.listdir(watertight_mesh_dir)

for watertight_mesh in watertight_meshes:
    watertight_mesh_name = watertight_mesh.split('.obj')[0]
    print(f'Processing {watertight_mesh_name}....')

    # Make masked object image
    image = cv2.imread(os.path.join(image_dir, f'{watertight_mesh_name}.jpg'))
    mask = cv2.imread(os.path.join(mask_dir, f'{watertight_mesh_name}.jpg'))[..., 0:1] > 128
    masked_obj_image = image * mask
    cv2.imwrite(os.path.join(masked_obj_image_dir, f'{watertight_mesh_name}.jpg'), masked_obj_image)

    # Run Paint3D
    os.system(f'python pipeline_paint3d_stage1.py  --sd_config controlnet/config/depth_based_inpaint_template.yaml  --render_config paint3d/config/train_config_paint3d.py  --mesh_path {watertight_mesh_dir}/{watertight_mesh_name}.obj --ip_adapter_image_path {masked_obj_image_dir}/{watertight_mesh_name}.jpg  --outdir {stage1_output_dir}/{watertight_mesh_name}')
    os.system(f'python pipeline_paint3d_stage2.py --sd_config controlnet/config/UV_based_inpaint_template.yaml  --render_config paint3d/config/train_config_paint3d.py  --mesh_path {watertight_mesh_dir}/{watertight_mesh_name}.obj --texture_path {stage1_output_dir}/{watertight_mesh_name}/res-0/albedo.png  --outdir {stage2_output_dir}/{watertight_mesh_name}')

    os.system('rm -rf paint3d_cache')