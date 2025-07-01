import cv2
import numpy as np
import trimesh
import torch
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import os
from PIL import Image
import matplotlib.pyplot as plt
from mesh_estimator import HumanMeshEstimator
from core.camerahmr_model import CameraHMR
from core.constants import CHECKPOINT_PATH, CAM_MODEL_CKPT, SMPL_MODEL_PATH, DETECTRON_CKPT, DETECTRON_CFG
from core.datasets.dataset import Dataset
from core.utils.renderer_pyrd import Renderer
from core.utils import recursive_to
from core.utils.geometry import batch_rot2aa
from core.cam_model.fl_net import FLNet
from core.constants import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD, NUM_BETAS
import moderngl
import pyrr
from scipy.spatial import KDTree

class ImprovedChestLogo3D:
    def __init__(self, smpl_model):
        self.smpl_model = smpl_model
        self.faces = smpl_model.faces
        self.chest_vertex_indices = self._get_anatomically_correct_chest_vertices()
        
    def _get_anatomically_correct_chest_vertices(self):
        """Get anatomically correct chest area vertex indices for SMPL model"""
        # More accurate chest vertices based on SMPL topology
        # These correspond to the actual chest/torso region in SMPL
        chest_vertices = []
        
        # Upper chest (clavicle to mid-chest)
        chest_vertices.extend([1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 1859])
        chest_vertices.extend([4870, 4871, 4872, 4873, 4874, 4875, 4876, 4877, 4878, 4879])
        
        # Mid chest (nipple line area) - MAIN CHEST AREA FOR LOGO
        chest_vertices.extend([1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244])
        chest_vertices.extend([4235, 4236, 4237, 4238, 4239, 4240, 4241, 4242, 4243, 4244])
        
        # Lower chest (below nipple line)
        chest_vertices.extend([3065, 3066, 3067, 3068, 3069, 3070, 3071, 3072, 3073, 3074])
        chest_vertices.extend([6065, 6066, 6067, 6068, 6069, 6070, 6071, 6072, 6073, 6074])
        
        # Central chest area for logo placement (MOST IMPORTANT)
        chest_vertices.extend([1200, 1201, 1202, 1203, 1204, 1205])
        chest_vertices.extend([4200, 4201, 4202, 4203, 4204, 4205])
        
        return chest_vertices
    
    def validate_chest_vertices(self, vertices):
        """Validate that chest vertices are anatomically correct"""
        valid_indices = [idx for idx in self.chest_vertex_indices if idx < len(vertices)]
        
        if len(valid_indices) < 10:
            print(f"Warning: Only {len(valid_indices)} valid chest vertices found")
            return False
        
        chest_vertices = vertices[valid_indices]
        
        # Check if vertices form a reasonable chest region
        # Chest should be on the front of the body (positive Z in SMPL coordinate system)
        avg_z = np.mean(chest_vertices[:, 2])
        if avg_z < 0:
            print("Warning: Chest vertices appear to be on the back of the body")
            return False
        
        # Check chest width and height are reasonable
        chest_width = np.max(chest_vertices[:, 0]) - np.min(chest_vertices[:, 0])
        chest_height = np.max(chest_vertices[:, 1]) - np.min(chest_vertices[:, 1])
        
        if chest_width < 0.1 or chest_height < 0.1:
            print(f"Warning: Chest region too small - width: {chest_width:.3f}, height: {chest_height:.3f}")
            return False
        
        return True
    
    def get_chest_region_info(self, vertices):
        """Get chest region center, normal, and local coordinate system with improved positioning"""
        # Filter valid indices
        valid_indices = [idx for idx in self.chest_vertex_indices if idx < len(vertices)]
        
        if not valid_indices or not self.validate_chest_vertices(vertices):
            # Improved fallback using body center estimation
            print("Using fallback chest position estimation")
            body_center = np.mean(vertices, axis=0)
            # FIXED: Better chest positioning - move down for proper chest level
            chest_center = body_center + np.array([0.0, 0.05, 0.05])  # Slightly down and forward
            chest_normal = np.array([0.0, 0.0, 1.0])
            right_vector = np.array([1.0, 0.0, 0.0])
            up_vector = np.array([0.0, 1.0, 0.0])
            return chest_center, chest_normal, right_vector, up_vector
        
        chest_vertices = vertices[valid_indices]
        
        # FIXED: Focus on mid-chest vertices for better logo positioning
        mid_chest_indices = [idx for idx in self.chest_vertex_indices[20:40] if idx < len(vertices)]
        if mid_chest_indices:
            mid_chest_vertices = vertices[mid_chest_indices]
            chest_center = np.mean(mid_chest_vertices, axis=0)
            # Adjust chest center to be at proper height
            chest_center[1] -= 0.03  # Move down slightly for better positioning
        else:
            chest_center = np.mean(chest_vertices, axis=0)
            chest_center[1] -= 0.05  # Move down more if using all chest vertices
        
        # Improved normal calculation using robust PCA
        centered_vertices = chest_vertices - chest_center
        
        # Remove outliers using median absolute deviation
        distances = np.linalg.norm(centered_vertices, axis=1)
        median_dist = np.median(distances)
        mad = np.median(np.abs(distances - median_dist))
        outlier_threshold = median_dist + 3 * mad
        
        inlier_mask = distances < outlier_threshold
        if np.sum(inlier_mask) > 5:  # Need at least 5 points
            centered_vertices = centered_vertices[inlier_mask]
        
        # Calculate normal using SVD for better stability
        U, S, Vt = np.linalg.svd(centered_vertices)
        chest_normal = Vt[-1]  # Last row of Vt is the normal
        
        # Ensure normal points outward (positive Z direction for front of body)
        if chest_normal[2] < 0:
            chest_normal = -chest_normal
        
        # Create orthonormal coordinate system
        # Up vector (Y-axis in local space) - align with body up direction
        world_up = np.array([0.0, 1.0, 0.0])
        right_vector = np.cross(chest_normal, world_up)
        right_vector = right_vector / (np.linalg.norm(right_vector) + 1e-8)
        
        up_vector = np.cross(right_vector, chest_normal)
        up_vector = up_vector / (np.linalg.norm(up_vector) + 1e-8)
        
        # Ensure right vector points to the person's right (negative X in SMPL)
        if right_vector[0] > 0:
            right_vector = -right_vector
            up_vector = np.cross(right_vector, chest_normal)
            up_vector = up_vector / (np.linalg.norm(up_vector) + 1e-8)
        
        return chest_center, chest_normal, right_vector, up_vector
    
    def create_chest_focused_texture(self, vertices, faces, chest_center, right_vector, up_vector, logo_image, logo_size=0.15):
        """Create texture that focuses the logo specifically on the chest area"""
        # Get chest region vertices
        valid_chest_indices = [idx for idx in self.chest_vertex_indices if idx < len(vertices)]
        
        if not valid_chest_indices:
            print("Warning: No valid chest vertices found, using fallback")
            valid_chest_indices = list(range(min(100, len(vertices))))
        
        # Create base skin texture
        texture_size = 1024
        base_texture = np.full((texture_size, texture_size, 3), [210, 180, 140], dtype=np.uint8)
        
        # Resize logo maintaining aspect ratio
        logo_h, logo_w = logo_image.shape[:2]
        logo_aspect = logo_w / logo_h
        
        # Calculate logo size in pixels (logo_size is in 3D world units)
        logo_pixel_size = int(texture_size * logo_size * 0.8)  # Scale factor for better visibility
        
        if logo_aspect > 1:
            logo_width = logo_pixel_size
            logo_height = int(logo_pixel_size / logo_aspect)
        else:
            logo_height = logo_pixel_size
            logo_width = int(logo_pixel_size * logo_aspect)
        
        # Ensure minimum logo size
        logo_width = max(64, logo_width)
        logo_height = max(64, logo_height)
        
        logo_resized = cv2.resize(logo_image, (logo_width, logo_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Calculate chest-specific UV mapping
        uv_coords = self._create_chest_focused_uv_mapping(vertices, chest_center, right_vector, up_vector)
        
        # FIXED: Better chest center positioning in UV space
        chest_u, chest_v = 0.5, 0.45  # Moved down from 0.35 to 0.45 for better chest level
        
        start_x = int((chest_u - 0.5 * logo_width / texture_size) * texture_size)
        start_y = int((chest_v - 0.5 * logo_height / texture_size) * texture_size)
        
        # Ensure logo fits within texture bounds
        start_x = max(0, min(start_x, texture_size - logo_width))
        start_y = max(0, min(start_y, texture_size - logo_height))
        
        # Apply logo with proper alpha blending
        if logo_resized.shape[2] == 4:
            alpha = logo_resized[:, :, 3:4] / 255.0
            logo_rgb = logo_resized[:, :, :3]
        else:
            alpha = np.ones((logo_height, logo_width, 1))
            logo_rgb = logo_resized
        
        # Create smooth blending
        texture_region = base_texture[start_y:start_y+logo_height, start_x:start_x+logo_width]
        blended = (alpha * logo_rgb + (1 - alpha) * texture_region).astype(np.uint8)
        base_texture[start_y:start_y+logo_height, start_x:start_x+logo_width] = blended
        
        return base_texture, uv_coords, (start_x, start_y, logo_width, logo_height)
    
    def _create_chest_focused_uv_mapping(self, vertices, chest_center, right_vector, up_vector):
        """Create UV mapping that properly maps the chest area to texture coordinates"""
        # Transform all vertices to chest-centered coordinate system
        centered_vertices = vertices - chest_center
        
        # Project onto chest coordinate system
        u_coords = np.dot(centered_vertices, right_vector)
        v_coords = np.dot(centered_vertices, up_vector)
        
        # Get chest region bounds for proper scaling
        valid_chest_indices = [idx for idx in self.chest_vertex_indices if idx < len(vertices)]
        if valid_chest_indices:
            chest_vertices = vertices[valid_chest_indices]
            chest_centered = chest_vertices - chest_center
            chest_u = np.dot(chest_centered, right_vector)
            chest_v = np.dot(chest_centered, up_vector)
            
            u_range = max(np.max(chest_u) - np.min(chest_u), 0.1)
            v_range = max(np.max(chest_v) - np.min(chest_v), 0.1)
            
            # Center the chest region in UV space
            u_center = np.mean(chest_u)
            v_center = np.mean(chest_v)
        else:
            # Fallback values
            u_range = max(np.max(u_coords) - np.min(u_coords), 0.1)
            v_range = max(np.max(v_coords) - np.min(v_coords), 0.1)
            u_center = np.mean(u_coords)
            v_center = np.mean(v_coords)
        
        # Normalize UV coordinates to [0,1] with chest centered
        u_normalized = (u_coords - u_center) / (u_range * 2) + 0.5
        v_normalized = (v_coords - v_center) / (v_range * 2) + 0.5
        
        # Clamp to valid UV range
        u_normalized = np.clip(u_normalized, 0.0, 1.0)
        v_normalized = np.clip(v_normalized, 0.0, 1.0)
        
        return np.column_stack([u_normalized, v_normalized])

class Enhanced3DVisualization:
    """Enhanced visualization with proper textured mesh rendering"""
    
    def __init__(self, width=1024, height=768):
        self.width = width
        self.height = height
        
    def render_textured_mesh_to_image(self, mesh_vertices, mesh_faces, texture_image, uv_coords, 
                                    camera_intrinsics, camera_translation, image_shape):
        """Render textured 3D mesh to 2D image coordinates with proper texture sampling"""
        h, w = image_shape[:2]
        rendered_image = np.zeros((h, w, 3), dtype=np.uint8)
        z_buffer = np.full((h, w), float('inf'))
        
        # Transform vertices to camera space
        vertices_cam = mesh_vertices + camera_translation
        
        # Project to image coordinates
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
        
        # Perspective projection
        vertices_2d = np.zeros((len(vertices_cam), 2))
        depths = vertices_cam[:, 2]
        
        # Avoid division by zero
        valid_depth_mask = depths > 0.01
        
        vertices_2d[valid_depth_mask, 0] = (vertices_cam[valid_depth_mask, 0] * fx / depths[valid_depth_mask]) + cx
        vertices_2d[valid_depth_mask, 1] = (vertices_cam[valid_depth_mask, 1] * fy / depths[valid_depth_mask]) + cy
        
        # Render each triangle with texture
        for face_idx, face in enumerate(mesh_faces):
            if not all(valid_depth_mask[idx] for idx in face):
                continue
                
            # Get triangle vertices in 2D
            triangle_2d = vertices_2d[face].astype(np.int32)
            triangle_depths = depths[face]
            triangle_uvs = uv_coords[face]
            
            # Check if triangle is within image bounds
            if (np.any(triangle_2d[:, 0] < 0) or np.any(triangle_2d[:, 0] >= w) or
                np.any(triangle_2d[:, 1] < 0) or np.any(triangle_2d[:, 1] >= h)):
                continue
            
            # Rasterize triangle
            self._rasterize_textured_triangle(
                rendered_image, z_buffer, triangle_2d, triangle_depths, 
                triangle_uvs, texture_image
            )
        
        return rendered_image
    
    def _rasterize_textured_triangle(self, image, z_buffer, triangle_2d, depths, uvs, texture):
        """Rasterize a single textured triangle using barycentric coordinates"""
        # Get triangle bounding box
        min_x = max(0, np.min(triangle_2d[:, 0]))
        max_x = min(image.shape[1] - 1, np.max(triangle_2d[:, 0]))
        min_y = max(0, np.min(triangle_2d[:, 1]))
        max_y = min(image.shape[0] - 1, np.max(triangle_2d[:, 1]))
        
        if min_x >= max_x or min_y >= max_y:
            return
        
        # Triangle vertices
        v0, v1, v2 = triangle_2d
        uv0, uv1, uv2 = uvs
        d0, d1, d2 = depths
        
        # Precompute triangle area for barycentric coordinates
        denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
        if abs(denom) < 1e-8:
            return
        
        texture_h, texture_w = texture.shape[:2]
        
        # Scan triangle
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                # Calculate barycentric coordinates
                w0 = ((v1[1] - v2[1]) * (x - v2[0]) + (v2[0] - v1[0]) * (y - v2[1])) / denom
                w1 = ((v2[1] - v0[1]) * (x - v2[0]) + (v0[0] - v2[0]) * (y - v2[1])) / denom
                w2 = 1 - w0 - w1
                
                # Check if point is inside triangle
                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    # Interpolate depth
                    depth = w0 * d0 + w1 * d1 + w2 * d2
                    
                    # Depth test
                    if depth < z_buffer[y, x]:
                        z_buffer[y, x] = depth
                        
                        # Interpolate UV coordinates
                        u = w0 * uv0[0] + w1 * uv1[0] + w2 * uv2[0]
                        v = w0 * uv0[1] + w1 * uv1[1] + w2 * uv2[1]
                        
                        # Sample texture
                        tex_x = int(np.clip(u * texture_w, 0, texture_w - 1))
                        tex_y = int(np.clip(v * texture_h, 0, texture_h - 1))
                        
                        image[y, x] = texture[tex_y, tex_x]
    
    def create_textured_mesh_overlay(self, original_image, mesh_vertices, mesh_faces, texture_image, 
                                   uv_coords, camera_intrinsics, camera_translation, blend_alpha=0.7):
        """Create overlay of textured 3D mesh on original image with FIXED color channels"""
        # Render the textured mesh
        rendered_mesh = self.render_textured_mesh_to_image(
            mesh_vertices, mesh_faces, texture_image, uv_coords,
            camera_intrinsics, camera_translation, original_image.shape
        )
        
        # FIXED: Convert rendered mesh from RGB to BGR to match OpenCV format
        rendered_mesh_bgr = cv2.cvtColor(rendered_mesh, cv2.COLOR_RGB2BGR)
        
        # Create mask where mesh was rendered (non-black pixels)
        mesh_mask = np.any(rendered_mesh_bgr > 0, axis=2)
        
        # Create blended overlay
        overlay_image = original_image.copy()
        
        # Blend rendered mesh with original image
        for c in range(3):
            overlay_image[mesh_mask, c] = (
                blend_alpha * rendered_mesh_bgr[mesh_mask, c] + 
                (1 - blend_alpha) * original_image[mesh_mask, c]
            ).astype(np.uint8)
        
        # FIXED: Return both BGR versions for consistency
        return overlay_image, rendered_mesh_bgr
    
    def create_texture_visualization(self, texture_image, logo_region_bounds=None):
        """Create visualization of the texture with logo highlighted"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original texture
        axes[0].imshow(texture_image)
        axes[0].set_title('Complete Texture with Logo')
        axes[0].axis('off')
        
        # Highlighted logo region
        if logo_region_bounds is not None:
            highlighted = texture_image.copy()
            x, y, w, h = logo_region_bounds
            # Draw rectangle around logo
            cv2.rectangle(highlighted, (x, y), (x + w, y + h), (255, 255, 0), 3)
            axes[1].imshow(highlighted)
            axes[1].set_title('Texture with Logo Region Highlighted')
        else:
            axes[1].imshow(texture_image)
            axes[1].set_title('Texture (No Logo Bounds)')
        axes[1].axis('off')
        
        plt.tight_layout()
        return fig
    
    def save_all_visualizations(self, output_folder, base_name, original_image, mesh_with_texture, 
                          overlay_image, texture_image, rendered_mesh=None, logo_bounds=None, combined_overlay=None):
        """Save all visualization formats with FIXED color handling"""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Save original image
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_original.jpg"), original_image)
        
        # Save overlay image
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_textured_overlay.jpg"), overlay_image)
        
        # FIXED: Save texture with proper color conversion
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_texture.png"), 
                   cv2.cvtColor(texture_image, cv2.COLOR_RGB2BGR))
        
        # Save 3D mesh
        if mesh_with_texture:
            mesh_with_texture.export(os.path.join(output_folder, f"{base_name}_mesh_with_logo.obj"))
        
        # FIXED: Save rendered mesh with proper color handling
        if rendered_mesh is not None:
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_rendered_mesh.png"), rendered_mesh)

        # ADD THIS SECTION HERE:
        if combined_overlay is not None:
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_combined_all_persons.jpg"), combined_overlay)

        # Create texture visualization  # This existing line
        texture_viz = self.create_texture_visualization(texture_image, logo_bounds)
        texture_viz.savefig(os.path.join(output_folder, f"{base_name}_texture_analysis.png"), 
                           dpi=150, bbox_inches='tight')
        plt.close(texture_viz)
        
        print(f"All visualizations saved to: {output_folder}")
        print(f"Files saved:")
        print(f"  - {base_name}_original.jpg (Original image)")
        print(f"  - {base_name}_textured_overlay.jpg (2D overlay with textured mesh)")
        print(f"  - {base_name}_texture.png (Applied texture)")
        print(f"  - {base_name}_mesh_with_logo.obj (3D mesh file)")
        print(f"  - {base_name}_rendered_mesh.png (Rendered textured mesh only)")
        print(f"  - {base_name}_texture_analysis.png (Texture analysis)")

class UltraEnhanced3DChestLogoHumanMeshEstimator(HumanMeshEstimator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chest_logo_3d = ImprovedChestLogo3D(self.smpl_model)
        self.visualizer = Enhanced3DVisualization()
    
    def load_and_validate_logo(self, logo_path):
        """Load and validate logo image with FIXED color handling"""
        try:
            logo_pil = Image.open(logo_path).convert("RGBA")
            logo_array = np.array(logo_pil)
            
            # Validate logo size
            if logo_array.shape[0] < 32 or logo_array.shape[1] < 32:
                print(f"Warning: Logo is very small ({logo_array.shape[1]}x{logo_array.shape[0]})")
            
            # FIXED: Proper color handling - keep RGB format for texture
            if logo_array.shape[2] == 4:
                alpha = logo_array[:, :, 3:4] / 255.0
                rgb = logo_array[:, :, :3]
                white_bg = np.full_like(rgb, 255)
                logo_rgb = (alpha * rgb + (1 - alpha) * white_bg).astype(np.uint8)
            else:
                logo_rgb = logo_array[:, :, :3]
            
            print(f"Logo loaded successfully: {logo_rgb.shape}")
            return logo_rgb
            
        except Exception as e:
            print(f"Error loading logo: {e}")
            return None
    
    def process_image_with_textured_3d_chest_logo(self, img_path, logo_path, output_img_folder, i, 
                                                logo_size=0.15):
        """Process image with textured 3D chest logo and proper overlay"""
        img_cv2 = cv2.imread(str(img_path))
        
        # Load and validate logo
        logo_image = self.load_and_validate_logo(logo_path)
        if logo_image is None:
            print(f"Could not load logo image: {logo_path}")
            return
        
        fname, img_ext = os.path.splitext(os.path.basename(img_path))
        
        # Detect humans in the image
        det_out = self.detector(img_cv2)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        
        if not valid_idx.any():
            print(f"No humans detected in {img_path}")
            return
        
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        bbox_scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0 
        bbox_center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0

        # Get Camera intrinsics
        cam_int = self.get_cam_intrinsics(img_cv2)
        dataset = Dataset(img_cv2, bbox_center, bbox_scale, cam_int, False, img_path)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

        person_count = 0
        # Add this line after person_count = 0
        combined_overlay = img_cv2.copy()  # Initialize combined overlay with original image
        
        # Process all batches and all persons
        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            img_h, img_w = batch['img_size'][0]
            
            with torch.no_grad():
                out_smpl_params, out_cam, focal_length_ = self.model(batch)

            output_vertices, output_joints, output_cam_trans = self.get_output_mesh(out_smpl_params, out_cam, batch)
            
            # Process each person in the batch
            for person_idx in range(output_vertices.shape[0]):
                vertices_np = output_vertices[person_idx].cpu().numpy()
                cam_trans = output_cam_trans[person_idx].cpu().numpy()
                
                print(f"\nProcessing person {person_count + 1}")
                print(f"Mesh vertices shape: {vertices_np.shape}")
                print(f"Camera translation: {cam_trans}")
                
                # FIXED: Get chest region information with improved positioning
                chest_center, chest_normal, right_vector, up_vector = self.chest_logo_3d.get_chest_region_info(vertices_np)
                
                print(f"Chest center: {chest_center}")
                print(f"Chest normal: {chest_normal}")
                
                # Create chest-focused texture with logo
                texture_image, uv_coords, logo_bounds = self.chest_logo_3d.create_chest_focused_texture(
                    vertices_np, self.smpl_model.faces, chest_center, right_vector, up_vector, 
                    logo_image, logo_size
                )
                
                # Create mesh with texture
                mesh = trimesh.Trimesh(vertices_np, self.smpl_model.faces, process=False)
                mesh.visual = trimesh.visual.TextureVisuals(
                    uv=uv_coords,
                    image=Image.fromarray(texture_image)
                )
                
                # # FIXED: Create textured mesh overlay with proper color handling
                # overlay_image, rendered_mesh = self.visualizer.create_textured_mesh_overlay(
                #     img_cv2, vertices_np, self.smpl_model.faces, texture_image, uv_coords,
                #     cam_int, cam_trans, blend_alpha=0.8
                # )
                
                # Create individual overlay for this person
                overlay_image, rendered_mesh = self.visualizer.create_textured_mesh_overlay(
                    img_cv2, vertices_np, self.smpl_model.faces, texture_image, uv_coords,
                    cam_int, cam_trans, blend_alpha=0.8
                )

                # Update combined overlay by blending this person's mesh onto it
                combined_overlay, _ = self.visualizer.create_textured_mesh_overlay(
                    combined_overlay, vertices_np, self.smpl_model.faces, texture_image, uv_coords,
                    cam_int, cam_trans, blend_alpha=0.8
                )
                
                # Generate base name for this person
                base_name = f"{fname}_{i:06d}_person_{person_count}"
                
                # Save all visualizations
                self.visualizer.save_all_visualizations(
                    output_img_folder, base_name, img_cv2, mesh, overlay_image, 
                    texture_image, rendered_mesh, logo_bounds, combined_overlay
                )
                
                person_count += 1
                
                print(f"Completed processing person {person_count} with textured 3D chest logo overlay")
        if person_count > 0:
            combined_base_name = f"{fname}_{i:06d}_all_persons_combined"
            cv2.imwrite(os.path.join(output_img_folder, f"{combined_base_name}.jpg"), combined_overlay)
            print(f"Combined overlay with all {person_count} persons saved as: {combined_base_name}.jpg")
        print(f"Total persons processed: {person_count}")
    
    def run_textured_3d_chest_logo_pipeline(self, image_folder, logo_path, out_folder, logo_size=0.15):
        """Run textured 3D chest logo pipeline with proper overlay"""
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        
        from glob import glob
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']
        images_list = [image for ext in image_extensions for image in glob(os.path.join(image_folder, ext))]
        
        print(f"=== TEXTURED 3D CHEST LOGO PIPELINE ===")
        print(f"Using logo: {logo_path}")
        print(f"Logo size: {logo_size}")
        print(f"Output folder: {out_folder}")
        print(f"Found {len(images_list)} images to process")
        
        for ind, img_path in enumerate(images_list):
            print(f"\n{'='*60}")
            print(f"Processing image {ind+1}/{len(images_list)}: {img_path}")
            print(f"{'='*60}")
            
            self.process_image_with_textured_3d_chest_logo(
                img_path, logo_path, out_folder, ind, logo_size=logo_size
            )
        
        print(f"\n{'='*60}")
        print("TEXTURED 3D CHEST LOGO PIPELINE COMPLETED!")
        print(f"All results saved to: {out_folder}")
        print(f"{'='*60}")

# Usage example with enhanced textured overlay
if __name__ == "__main__":
    import argparse
    from glob import glob
    
    def make_parser():
        parser = argparse.ArgumentParser(description='Enhanced CameraHMR with Textured 3D Chest Logo Overlay')
        parser.add_argument("--image_folder", type=str, required=True, help="Path to input image folder.")
        parser.add_argument("--logo_path", type=str, required=True, help="Path to logo image.")
        parser.add_argument("--output_folder", type=str, required=True, help="Path to output folder.")
        parser.add_argument("--logo_size", type=float, default=1.0, help="Logo size on mesh (0.1 = small, 0.3 = large)")
        parser.add_argument("--blend_alpha", type=float, default=0.8, help="Blending alpha for overlay (0.0 = original image, 1.0 = full mesh)")
        return parser

    parser = make_parser()
    args = parser.parse_args()
    
    # Use ultra-enhanced 3D chest logo estimator with textured overlay
    estimator = UltraEnhanced3DChestLogoHumanMeshEstimator()
    estimator.run_textured_3d_chest_logo_pipeline(
        args.image_folder, 
        args.logo_path, 
        args.output_folder, 
        args.logo_size
    )