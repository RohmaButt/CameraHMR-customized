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
from scipy.interpolate import griddata
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from functools import lru_cache

class DynamicChestLogo3D:
    def __init__(self, smpl_model):
        self.smpl_model = smpl_model
        self.faces = smpl_model.faces
        
        # SMPL joint indices
        self.joint_indices = {
            'pelvis': 0,
            'left_hip': 1,
            'right_hip': 2,
            'spine1': 3,
            'spine2': 6,
            'spine3': 9,
            'neck': 12,
            'left_collar': 13,
            'right_collar': 14,
            'head': 15,
            'left_shoulder': 16,
            'right_shoulder': 17
        }
        
        # Cache for frequently used computations
        self._cached_chest_vertices = None
        self._cached_face_adjacency = None
        
    def get_dynamic_chest_coordinates(self, vertices, joints_3d):
        """
        Dynamically compute chest coordinates using SMPL joints
        
        Args:
            vertices: SMPL mesh vertices
            joints_3d: SMPL 3D joint positions
            
        Returns:
            chest_center, chest_normal, right_vector, up_vector
        """
        try:
            # Extract key joint positions
            left_shoulder = joints_3d[self.joint_indices['left_shoulder']]
            right_shoulder = joints_3d[self.joint_indices['right_shoulder']]
            spine2 = joints_3d[self.joint_indices['spine2']]  # mid spine
            spine3 = joints_3d[self.joint_indices['spine3']]  # upper spine
            neck = joints_3d[self.joint_indices['neck']]
            
            # Optional: use collar bones if available (more accurate for chest)
            try:
                left_collar = joints_3d[self.joint_indices['left_collar']]
                right_collar = joints_3d[self.joint_indices['right_collar']]
                shoulder_midpoint = (left_collar + right_collar) / 2.0
            except (IndexError, KeyError):
                # Fallback to shoulders if collar bones not available
                shoulder_midpoint = (left_shoulder + right_shoulder) / 2.0
            
            # Calculate chest center
            # Position slightly below the shoulder line and above mid-spine
            chest_vertical_ratio = 0.3  # 30% down from shoulders towards spine2
            chest_center = shoulder_midpoint + chest_vertical_ratio * (spine2 - shoulder_midpoint)
            
            # Move chest center slightly forward (towards camera/front of body)
            # Calculate body orientation
            shoulder_vector = right_shoulder - left_shoulder
            spine_vector = neck - spine2
            
            # Calculate forward direction (perpendicular to shoulder line and spine)
            forward_direction = np.cross(shoulder_vector, spine_vector)
            forward_direction = forward_direction / (np.linalg.norm(forward_direction) + 1e-8)
            
            # Ensure forward direction points outward (positive Z in camera space typically)
            if forward_direction[2] < 0:
                forward_direction = -forward_direction
            
            # Move chest center forward by a small amount
            chest_offset = 0.05  # 5cm forward
            chest_center = chest_center + chest_offset * forward_direction
            
            # Calculate coordinate system
            chest_normal = forward_direction
            
            # Right vector (from left shoulder to right shoulder, normalized)
            right_vector = (right_shoulder - left_shoulder)
            right_vector = right_vector / (np.linalg.norm(right_vector) + 1e-8)
            
            # Up vector (perpendicular to both normal and right)
            up_vector = np.cross(chest_normal, right_vector)
            up_vector = up_vector / (np.linalg.norm(up_vector) + 1e-8)
            
            # Ensure consistent orientation
            if up_vector[1] < 0:  # Y should generally point up
                up_vector = -up_vector
                chest_normal = np.cross(right_vector, up_vector)
                chest_normal = chest_normal / (np.linalg.norm(chest_normal) + 1e-8)
            
            return chest_center, chest_normal, right_vector, up_vector
            
        except Exception as e:
            print(f"Error in dynamic chest calculation: {e}")
            return self._get_fallback_chest_info(vertices)
    
    def get_chest_vertices_from_joints(self, vertices, joints_3d):
        """
        Find chest vertices based on joint positions using spatial proximity
        """
        chest_center, chest_normal, right_vector, up_vector = self.get_dynamic_chest_coordinates(vertices, joints_3d)
        
        # Define chest region boundaries based on joint positions
        left_shoulder = joints_3d[self.joint_indices['left_shoulder']]
        right_shoulder = joints_3d[self.joint_indices['right_shoulder']]
        spine2 = joints_3d[self.joint_indices['spine2']]
        
        # Calculate chest region dimensions
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        chest_width = shoulder_width * 0.8  # Chest is slightly narrower than shoulders
        chest_height = np.linalg.norm(joints_3d[self.joint_indices['neck']] - spine2) * 0.6
        
        # Find vertices within chest region
        chest_vertices = []
        
        for i, vertex in enumerate(vertices):
            # Transform vertex to chest coordinate system
            relative_pos = vertex - chest_center
            
            # Project onto chest coordinate system
            u_coord = np.dot(relative_pos, right_vector)
            v_coord = np.dot(relative_pos, up_vector)
            depth_coord = np.dot(relative_pos, chest_normal)
            
            # Check if vertex is within chest region
            if (abs(u_coord) <= chest_width / 2 and 
                abs(v_coord) <= chest_height / 2 and 
                depth_coord >= -0.05 and depth_coord <= 0.1):  # Front-facing vertices
                chest_vertices.append(i)
        
        return chest_vertices
    
    def validate_chest_vertices_dynamic(self, vertices, joints_3d):
        """Enhanced validation for dynamically found chest vertices"""
        try:
            chest_vertices_indices = self.get_chest_vertices_from_joints(vertices, joints_3d)
            
            if len(chest_vertices_indices) < 20:
                return False
            
            chest_vertices = vertices[chest_vertices_indices]
            
            # Check if vertices form a reasonable chest region
            chest_ranges = np.ptp(chest_vertices, axis=0)
            chest_width, chest_height = chest_ranges[0], chest_ranges[1]
            
            if chest_width < 0.05 or chest_height < 0.05:
                return False
            
            # Check if chest vertices are generally front-facing
            chest_center, chest_normal, _, _ = self.get_dynamic_chest_coordinates(vertices, joints_3d)
            avg_depth = np.mean([np.dot(v - chest_center, chest_normal) for v in chest_vertices])
            
            if avg_depth < -0.1:  # Too far back
                return False
            
            return True
            
        except Exception as e:
            print(f"Validation error: {e}")
            return False
    
    def get_chest_region_info_dynamic(self, vertices, joints_3d):
        """Get chest region info using dynamic joint-based calculation"""
        try:
            if not self.validate_chest_vertices_dynamic(vertices, joints_3d):
                print("Using fallback chest calculation")
                return self._get_fallback_chest_info(vertices)
            
            return self.get_dynamic_chest_coordinates(vertices, joints_3d)
            
        except Exception as e:
            print(f"Error in dynamic chest region calculation: {e}")
            return self._get_fallback_chest_info(vertices)
    
    def _get_fallback_chest_info(self, vertices):
        """Enhanced fallback chest positioning"""
        # Vectorized body center calculation
        body_center = np.mean(vertices, axis=0)
        
        # Estimate chest position relative to body center
        y_range = np.ptp(vertices[:, 1])
        chest_offset_y = -0.15 * y_range  # Slightly above center
        
        chest_center = body_center + np.array([0.0, chest_offset_y, 0.08])
        chest_normal = np.array([0.0, 0.0, 1.0])
        right_vector = np.array([-1.0, 0.0, 0.0])
        up_vector = np.array([0.0, 1.0, 0.0])
        
        return chest_center, chest_normal, right_vector, up_vector
    
    def load_and_validate_logo_cached(self, logo_path):
        """Load logo with automatic white background removal and transparency detection"""
        try:
            logo_pil = Image.open(logo_path).convert("RGBA")
            logo_array = np.array(logo_pil)
            
            if logo_array.shape[0] < 32 or logo_array.shape[1] < 32:
                print(f"Warning: Logo is very small ({logo_array.shape[1]}x{logo_array.shape[0]})")
            
            # Enhanced: Automatic white background removal
            logo_array = self._remove_white_background(logo_array)
            
            print(f"Logo loaded with enhanced transparency processing: {logo_array.shape}")
            return logo_array
            
        except Exception as e:
            print(f"Error loading logo: {e}")
            return None

    def _remove_white_background(self, logo_array):
        """Remove white/near-white backgrounds and convert to transparent"""
        rgb = logo_array[:, :, :3]
        alpha = logo_array[:, :, 3] if logo_array.shape[2] == 4 else np.full(rgb.shape[:2], 255, dtype=np.uint8)
        
        # Define white/near-white threshold
        white_threshold = 240
        
        # Create mask for white/near-white pixels
        white_mask = np.all(rgb >= white_threshold, axis=2)
        
        # Also check for light gray backgrounds
        light_gray_threshold = 230
        light_gray_mask = np.all(rgb >= light_gray_threshold, axis=2) & np.all(np.abs(rgb - np.mean(rgb, axis=2, keepdims=True)) < 10, axis=2)
        
        # Combine masks
        background_mask = white_mask | light_gray_mask
        
        # Set alpha to 0 for background pixels
        alpha[background_mask] = 0
        
        # For edge smoothing, create a gradient around the edges
        from scipy import ndimage
        
        # Create distance transform from non-background pixels
        foreground_mask = ~background_mask
        if np.any(foreground_mask):
            # Dilate the foreground slightly to create anti-aliased edges
            dilated = ndimage.binary_dilation(foreground_mask, iterations=2)
            edge_mask = dilated & ~foreground_mask
            
            # Create gradient alpha for edge pixels
            if np.any(edge_mask):
                alpha[edge_mask] = 128  # Semi-transparent edges
        
        # Combine RGB and alpha
        result = np.dstack([rgb, alpha])
        
        print(f"Background removal: {np.sum(background_mask)} pixels made transparent")
        return result

    def _get_correctly_oriented_logo(self, logo_image, right_vector, up_vector):
        """Determine correct logo orientation based on mesh coordinate system"""
        # Analyze the coordinate system orientation
        if right_vector[0] > 0:
            # Right vector points right, normal orientation
            if up_vector[1] > 0:
                return logo_image
            else:
                return cv2.flip(logo_image, 0)
        else:
            # Right vector points left, we need horizontal flip
            if up_vector[1] > 0:
                return cv2.flip(logo_image, 1)
            else:
                return cv2.rotate(logo_image, cv2.ROTATE_180)
    
    def create_chest_focused_texture_dynamic(self, vertices, faces, joints_3d, logo_image, logo_size=0.15):
        """Create texture with dynamic chest positioning based on joints - UPDATED with MAGENTA base"""
        # Pre-allocate arrays - CHANGED to bright magenta for easy removal
        texture_size = 1024
        base_texture = np.full((texture_size, texture_size, 3), [255, 0, 255], dtype=np.uint8)  # Bright magenta
        
        # Get dynamic chest coordinates
        chest_center, chest_normal, right_vector, up_vector = self.get_chest_region_info_dynamic(vertices, joints_3d)
        
        # Dynamic logo orientation correction
        logo_corrected = self._get_correctly_oriented_logo(logo_image, right_vector, up_vector)
        
        # Calculate logo dimensions
        logo_h, logo_w = logo_corrected.shape[:2]
        logo_aspect = logo_w / logo_h
        logo_pixel_size = int(texture_size * logo_size * 0.6)
        
        if logo_aspect > 1:
            logo_width = logo_pixel_size
            logo_height = int(logo_pixel_size / logo_aspect)
        else:
            logo_height = logo_pixel_size
            logo_width = int(logo_pixel_size * logo_aspect)
        
        logo_width = max(48, logo_width)
        logo_height = max(48, logo_height)
        
        # Resize logo
        logo_resized = cv2.resize(logo_corrected, (logo_width, logo_height), interpolation=cv2.INTER_LINEAR)
        
        # Create UV mapping based on dynamic chest coordinates
        uv_coords = self._create_dynamic_uv_mapping(vertices, faces, chest_center, right_vector, up_vector, chest_normal, joints_3d)
        
        # Logo placement (centered on chest)
        chest_u, chest_v = 0.5, 0.55
        start_x = int((chest_u - 0.5 * logo_width / texture_size) * texture_size)
        start_y = int((chest_v - 0.5 * logo_height / texture_size) * texture_size)
        
        start_x = max(0, min(start_x, texture_size - logo_width))
        start_y = max(0, min(start_y, texture_size - logo_height))
        
        # Enhanced alpha blending
        if logo_resized.shape[2] == 4:
            alpha = logo_resized[:, :, 3].astype(np.float32) / 255.0
            logo_rgb = logo_resized[:, :, :3]
            
            # Get the texture region
            texture_region = base_texture[start_y:start_y+logo_height, start_x:start_x+logo_width].astype(np.float32)
            logo_rgb_float = logo_rgb.astype(np.float32)
            
            # Apply alpha blending
            alpha_3d = np.stack([alpha] * 3, axis=2)
            blended = alpha_3d * logo_rgb_float + (1 - alpha_3d) * texture_region
            
            base_texture[start_y:start_y+logo_height, start_x:start_x+logo_width] = blended.astype(np.uint8)
        else:
            # For RGB logos without alpha, try to detect and remove white backgrounds
            logo_rgb = logo_resized
            
            # Create alpha channel by detecting white/near-white pixels
            white_threshold = 240
            white_mask = np.all(logo_rgb >= white_threshold, axis=2)
            
            # Create alpha channel
            alpha = np.ones(logo_rgb.shape[:2], dtype=np.float32)
            alpha[white_mask] = 0.0
            
            # Apply blending
            texture_region = base_texture[start_y:start_y+logo_height, start_x:start_x+logo_width].astype(np.float32)
            logo_rgb_float = logo_rgb.astype(np.float32)
            alpha_3d = np.stack([alpha] * 3, axis=2)
            
            blended = alpha_3d * logo_rgb_float + (1 - alpha_3d) * texture_region
            base_texture[start_y:start_y+logo_height, start_x:start_x+logo_width] = blended.astype(np.uint8)
        
        return base_texture, uv_coords, (start_x, start_y, logo_width, logo_height)
    
    def _create_dynamic_uv_mapping(self, vertices, faces, chest_center, right_vector, up_vector, chest_normal, joints_3d):
        """Create UV mapping using dynamic chest coordinates"""
        # Find chest vertices dynamically
        chest_vertices_indices = self.get_chest_vertices_from_joints(vertices, joints_3d)
        
        if not chest_vertices_indices:
            return self._create_fallback_uv_mapping(vertices, chest_center, right_vector, up_vector)
        
        # Calculate distances to chest center
        distances_to_chest = np.linalg.norm(vertices - chest_center, axis=1)
        chest_vertices = vertices[chest_vertices_indices]
        max_chest_distance = np.max(np.linalg.norm(chest_vertices - chest_center, axis=1))
        
        # Weight calculation
        chest_weights = np.exp(-distances_to_chest / (max_chest_distance * 0.5))
        
        # Transform to chest coordinate system
        centered_vertices = vertices - chest_center
        u_coords = np.dot(centered_vertices, right_vector)
        v_coords = np.dot(centered_vertices, up_vector)
        depth_coords = np.dot(centered_vertices, chest_normal)
        
        # Curvature correction
        curvature_correction = 1.0 + 0.2 * np.abs(depth_coords) / (max_chest_distance + 1e-8)
        
        # Calculate ranges based on actual chest vertices
        chest_centered = chest_vertices - chest_center
        chest_u = np.dot(chest_centered, right_vector)
        chest_v = np.dot(chest_centered, up_vector)
        
        u_range = max(np.ptp(chest_u), 0.1)
        v_range = max(np.ptp(chest_v), 0.1)
        
        u_center = np.mean(chest_u)
        v_center = np.mean(chest_v)
        
        # Normalize coordinates
        u_normalized = (u_coords - u_center) / (u_range * 2 * curvature_correction) + 0.5
        v_normalized = (v_coords - v_center) / (v_range * 2 * curvature_correction) + 0.5
        
        # Apply weighting
        target_u, target_v = 0.5, 0.55
        u_normalized = chest_weights * target_u + (1 - chest_weights) * u_normalized
        v_normalized = chest_weights * target_v + (1 - chest_weights) * v_normalized
        
        # Clamp coordinates
        u_normalized = np.clip(u_normalized, 0.0, 1.0)
        v_normalized = np.clip(v_normalized, 0.0, 1.0)
        
        return np.column_stack([u_normalized, v_normalized])
    
    def _create_fallback_uv_mapping(self, vertices, chest_center, right_vector, up_vector):
        """Fallback UV mapping when dynamic method fails"""
        centered_vertices = vertices - chest_center
        u_coords = np.dot(centered_vertices, right_vector)
        v_coords = np.dot(centered_vertices, up_vector)
        
        # Normalize
        u_range = max(np.ptp(u_coords), 0.1)
        v_range = max(np.ptp(v_coords), 0.1)
        
        u_center = np.mean(u_coords)
        v_center = np.mean(v_coords)
        
        u_normalized = (u_coords - u_center) / (u_range * 2.5) + 0.5
        v_normalized = (v_coords - v_center) / (v_range * 2.5) + 0.4
        
        return np.column_stack([np.clip(u_normalized, 0.0, 1.0), np.clip(v_normalized, 0.0, 1.0)])

# NEW: Post-processing class for magenta removal and transparency creation
class PostProcessingTransparency:
    """Post-processing class to remove magenta base color and create transparency"""
    
    def __init__(self, magenta_color=[255, 0, 255], tolerance=30):
        self.magenta_color = np.array(magenta_color)  # Bright magenta RGB
        self.tolerance = tolerance  # Color tolerance for matching
    
    def create_magenta_mask(self, image):
        """Create mask for magenta pixels"""
        # Convert BGR to RGB if needed
        if image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Calculate color distance from magenta
        color_diff = np.linalg.norm(image_rgb - self.magenta_color, axis=2)
        
        # Create mask for pixels close to magenta
        magenta_mask = color_diff <= self.tolerance
        
        return magenta_mask
    
    def remove_magenta_create_transparent(self, overlay_image, rendered_mesh_image, original_image):
        """
        Remove magenta base color from overlay and create transparent version
        
        Args:
            overlay_image: Final overlay image (BGR)
            rendered_mesh_image: Pure rendered mesh (BGR) 
            original_image: Original input image (BGR)
            
        Returns:
            transparent_overlay: RGBA image with magenta made transparent
            clean_overlay: Cleaned overlay without magenta artifacts
        """
        
        # Create magenta mask from rendered mesh
        magenta_mask = self.create_magenta_mask(rendered_mesh_image)
        
        # Also check overlay image for any remaining magenta
        overlay_magenta_mask = self.create_magenta_mask(overlay_image)
        
        # Combine masks
        combined_magenta_mask = magenta_mask | overlay_magenta_mask
        
        # Create alpha channel
        alpha_channel = np.full((overlay_image.shape[0], overlay_image.shape[1]), 255, dtype=np.uint8)
        alpha_channel[combined_magenta_mask] = 0  # Make magenta pixels transparent
        
        # Create RGBA version
        overlay_rgb = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
        transparent_overlay = np.dstack([overlay_rgb, alpha_channel])
        
        # Create clean overlay by replacing magenta areas with original image
        clean_overlay = overlay_image.copy()
        clean_overlay[combined_magenta_mask] = original_image[combined_magenta_mask]
        
        # Additional cleanup: smooth edges around logo areas
        clean_overlay = self._smooth_magenta_edges(clean_overlay, combined_magenta_mask, original_image)
        
        return transparent_overlay, clean_overlay
    
    def _smooth_magenta_edges(self, image, magenta_mask, original_image, kernel_size=5):
        """Smooth edges around removed magenta areas"""
        from scipy import ndimage
        
        # Create edge mask by dilating magenta mask
        dilated_mask = ndimage.binary_dilation(magenta_mask, iterations=kernel_size//2)
        edge_mask = dilated_mask & ~magenta_mask
        
        if np.any(edge_mask):
            # Apply slight blur to edge pixels
            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            
            # Blend edge pixels with blurred version
            image[edge_mask] = (0.7 * image[edge_mask] + 0.3 * blurred[edge_mask]).astype(np.uint8)
        
        return image
    
    def save_transparent_results(self, transparent_overlay, clean_overlay, output_folder, base_name):
        """Save transparent and clean results"""
        # Save transparent RGBA version
        transparent_path = os.path.join(output_folder, f"{base_name}_transparent.png")
        transparent_pil = Image.fromarray(transparent_overlay, 'RGBA')
        transparent_pil.save(transparent_path, 'PNG')
        
        # Save clean overlay (BGR)
        clean_path = os.path.join(output_folder, f"{base_name}_clean_overlay.jpg")
        cv2.imwrite(clean_path, clean_overlay)
        
        print(f"Saved transparent version: {transparent_path}")
        print(f"Saved clean overlay: {clean_path}")
        
        return transparent_path, clean_path

# Enhanced visualization class with post-processing
class OcclusionAwareVisualization:
    """Enhanced visualization with proper depth testing and occlusion handling"""
    
    def __init__(self, width=1024, height=768):
        self.width = width
        self.height = height
        self.post_processor = PostProcessingTransparency()  # Add post-processor
        
    def render_multiple_meshes_with_occlusion(self, mesh_data_list, camera_intrinsics, image_shape):
        """
        Render multiple meshes with proper occlusion handling
        
        Args:
            mesh_data_list: List of tuples (vertices, faces, texture, uv_coords, cam_trans, person_id)
            camera_intrinsics: Camera intrinsic matrix
            image_shape: Output image shape
            
        Returns:
            rendered_image: Final rendered image with proper occlusion
            depth_map: Depth map for debugging
        """
        h, w = image_shape[:2]
        rendered_image = np.zeros((h, w, 3), dtype=np.uint8)
        z_buffer = np.full((h, w), float('inf'))
        person_id_buffer = np.full((h, w), -1, dtype=np.int32)  # Track which person owns each pixel
        
        # Collect all triangles from all meshes with their depths
        all_triangles = []
        
        for person_idx, (vertices, faces, texture, uv_coords, cam_trans, person_id) in enumerate(mesh_data_list):
            # Transform vertices to camera space
            vertices_cam = vertices + cam_trans
            
            # Project to 2D
            fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
            cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
            
            depths = vertices_cam[:, 2]
            valid_depth_mask = depths > 0.01
            
            # Project vertices
            vertices_2d = np.zeros((len(vertices_cam), 2))
            valid_indices = np.where(valid_depth_mask)[0]
            
            if len(valid_indices) > 0:
                vertices_2d[valid_indices, 0] = (vertices_cam[valid_indices, 0] * fx / depths[valid_indices]) + cx
                vertices_2d[valid_indices, 1] = (vertices_cam[valid_indices, 1] * fy / depths[valid_indices]) + cy
            
            # Process faces and create triangle data
            for face in faces:
                if all(valid_depth_mask[idx] for idx in face):
                    triangle_2d = vertices_2d[face].astype(np.int32)
                    triangle_depths = depths[face]
                    triangle_uvs = uv_coords[face]
                    
                    # Check if triangle is within image bounds
                    if not (np.any(triangle_2d[:, 0] < 0) or np.any(triangle_2d[:, 0] >= w) or
                           np.any(triangle_2d[:, 1] < 0) or np.any(triangle_2d[:, 1] >= h)):
                        
                        # Calculate triangle's average depth for sorting
                        avg_depth = np.mean(triangle_depths)
                        
                        all_triangles.append({
                            'triangle_2d': triangle_2d,
                            'depths': triangle_depths,
                            'uvs': triangle_uvs,
                            'texture': texture,
                            'avg_depth': avg_depth,
                            'person_id': person_id,
                            'face_indices': face
                        })
        
        # Sort triangles by depth (back to front for proper alpha blending)
        all_triangles.sort(key=lambda x: x['avg_depth'], reverse=True)
        
        # Render triangles with proper depth testing
        for triangle_data in all_triangles:
            self._rasterize_triangle_with_occlusion(
                rendered_image, z_buffer, person_id_buffer, triangle_data
            )
        
        return rendered_image, z_buffer, person_id_buffer
    
    def _rasterize_triangle_with_occlusion(self, image, z_buffer, person_id_buffer, triangle_data):
        """Rasterize triangle with proper depth testing and occlusion handling"""
        triangle_2d = triangle_data['triangle_2d']
        depths = triangle_data['depths']
        uvs = triangle_data['uvs']
        texture = triangle_data['texture']
        person_id = triangle_data['person_id']
        
        # Calculate bounding box
        min_x = max(0, np.min(triangle_2d[:, 0]))
        max_x = min(image.shape[1] - 1, np.max(triangle_2d[:, 0]))
        min_y = max(0, np.min(triangle_2d[:, 1]))
        max_y = min(image.shape[0] - 1, np.max(triangle_2d[:, 1]))
        
        if min_x >= max_x or min_y >= max_y:
            return
        
        v0, v1, v2 = triangle_2d
        uv0, uv1, uv2 = uvs
        d0, d1, d2 = depths
        
        # Precompute triangle area for barycentric coordinates
        denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
        if abs(denom) < 1e-8:
            return
        
        texture_h, texture_w = texture.shape[:2]
        
        # Rasterize with depth testing
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                # Calculate barycentric coordinates
                w0 = ((v1[1] - v2[1]) * (x - v2[0]) + (v2[0] - v1[0]) * (y - v2[1])) / denom
                w1 = ((v2[1] - v0[1]) * (x - v2[0]) + (v0[0] - v2[0]) * (y - v2[1])) / denom
                w2 = 1 - w0 - w1
                
                # Check if point is inside triangle
                if w0 < 0 or w1 < 0 or w2 < 0:
                    continue
                
                # Interpolate depth
                depth = w0 * d0 + w1 * d1 + w2 * d2
                
                # Depth test with small bias to handle floating point precision
                depth_bias = 1e-6
                if depth >= z_buffer[y, x] - depth_bias:
                    continue
                
                # Update depth buffer and person ID
                z_buffer[y, x] = depth
                person_id_buffer[y, x] = person_id
                
                # Interpolate UV coordinates
                u = w0 * uv0[0] + w1 * uv1[0] + w2 * uv2[0]
                v = w0 * uv0[1] + w1 * uv1[1] + w2 * uv2[1]
                
                # Sample texture
                tex_x = int(np.clip(u * texture_w, 0, texture_w - 1))
                tex_y = int(np.clip(v * texture_h, 0, texture_h - 1))
                
                # Apply texture
                sampled_color = texture[tex_y, tex_x]
                
                # Handle alpha channel if present
                if texture.shape[2] == 4:
                    alpha = sampled_color[3] / 255.0
                    if alpha > 0.1:  # Only render if alpha is significant
                        blended_color = alpha * sampled_color[:3] + (1 - alpha) * image[y, x]
                        image[y, x] = blended_color.astype(np.uint8)
                else:
                    image[y, x] = sampled_color[:3]

    def create_occlusion_aware_overlay(self, original_image, mesh_data_list, camera_intrinsics, 
                                     blend_alpha=0.7, crop_chest_bottom=True, 
                                     crop_bottom_percentage=0.5, crop_top_percentage=0.1):
        """
        Create overlay with proper occlusion handling between multiple people
        
        Args:
            original_image: Original input image
            mesh_data_list: List of mesh data for each person
            camera_intrinsics: Camera intrinsic matrix
            blend_alpha: Blending alpha for overlay
            crop_chest_bottom: Whether to crop bottom of chest
            crop_bottom_percentage: Percentage to crop from bottom
            crop_top_percentage: Percentage to crop from top
            
        Returns:
            overlay_image: Final overlaid image
            rendered_mesh: Pure rendered mesh image
            debug_info: Dictionary with debugging information
        """
        # Render all meshes with proper occlusion
        rendered_mesh, depth_map, person_id_map = self.render_multiple_meshes_with_occlusion(
            mesh_data_list, camera_intrinsics, original_image.shape
        )
        
        # Convert to BGR for OpenCV
        rendered_mesh_bgr = cv2.cvtColor(rendered_mesh, cv2.COLOR_RGB2BGR)
        
        # Create combined mask for all rendered content
        mesh_mask = np.any(rendered_mesh_bgr > 0, axis=2)
        
        # Apply chest cropping if requested
        if crop_chest_bottom:
            mesh_mask = self._apply_multi_person_chest_cropping(
                mesh_mask, mesh_data_list, camera_intrinsics, original_image.shape,
                crop_bottom_percentage, crop_top_percentage
            )
        
        # Create final overlay
        overlay_image = original_image.copy()
        overlay_image[mesh_mask] = (
            blend_alpha * rendered_mesh_bgr[mesh_mask] + 
            (1 - blend_alpha) * original_image[mesh_mask]
        ).astype(np.uint8)
        
        # Debug information
        debug_info = {
            'depth_map': depth_map,
            'person_id_map': person_id_map,
            'mesh_mask': mesh_mask,
            'num_people': len(mesh_data_list)
        }
        
        return overlay_image, rendered_mesh_bgr, debug_info

    def _apply_multi_person_chest_cropping(self, mesh_mask, mesh_data_list, camera_intrinsics, 
                                         image_shape, crop_bottom_percentage=0.5, crop_top_percentage=0.1):
        """Apply chest cropping for multiple people"""
        h, w = image_shape[:2]
        modified_mask = mesh_mask.copy()
        
        for person_idx, (vertices, faces, texture, uv_coords, cam_trans, person_id) in enumerate(mesh_data_list):
            # Transform vertices to camera space
            vertices_cam = vertices + cam_trans
            
            # Project to 2D
            fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
            cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
            
            depths = vertices_cam[:, 2]
            valid_depth_mask = depths > 0.01
            
            if not np.any(valid_depth_mask):
                continue
            
            # Project vertices
            vertices_2d = np.zeros((len(vertices_cam), 2))
            valid_indices = np.where(valid_depth_mask)[0]
            
            if len(valid_indices) > 0:
                vertices_2d[valid_indices, 0] = (vertices_cam[valid_indices, 0] * fx / depths[valid_indices]) + cx
                vertices_2d[valid_indices, 1] = (vertices_cam[valid_indices, 1] * fy / depths[valid_indices]) + cy
            
            # Find this person's projected area
            valid_2d_points = vertices_2d[valid_indices]
            valid_2d_points = valid_2d_points[
                (valid_2d_points[:, 0] >= 0) & (valid_2d_points[:, 0] < w) &
                (valid_2d_points[:, 1] >= 0) & (valid_2d_points[:, 1] < h)
            ]
            
            if len(valid_2d_points) == 0:
                continue
            
            # Calculate bounding box for this person
            min_x = int(np.min(valid_2d_points[:, 0]))
            max_x = int(np.max(valid_2d_points[:, 0]))
            min_y = int(np.min(valid_2d_points[:, 1]))
            max_y = int(np.max(valid_2d_points[:, 1]))
            
            # Apply cropping for this person's area
            person_height = max_y - min_y
            bottom_crop_line_y = max_y - int(person_height * crop_bottom_percentage)
            top_crop_line_y = min_y + int(person_height * crop_top_percentage)
            
            # Apply cropping to the mask
            if bottom_crop_line_y < max_y:
                modified_mask[bottom_crop_line_y:max_y + 1, min_x:max_x + 1] = False
            if top_crop_line_y > min_y:
                modified_mask[min_y:top_crop_line_y + 1, min_x:max_x + 1] = False
        
        return modified_mask

# Updated main estimator class with post-processing
class DynamicChestLogoHumanMeshEstimator(HumanMeshEstimator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chest_logo_3d = DynamicChestLogo3D(self.smpl_model)
        self.visualizer = OcclusionAwareVisualization()  # Use new occlusion-aware visualizer
        
        # Cache for logo
        self._cached_logo = None
        self._cached_logo_path = None
    
    def load_and_validate_logo_cached(self, logo_path):
        """Load and validate logo image with caching"""
        if self._cached_logo_path == logo_path and self._cached_logo is not None:
            return self._cached_logo
        
        logo_array = self.chest_logo_3d.load_and_validate_logo_cached(logo_path)
        if logo_array is not None:
            self._cached_logo = logo_array
            self._cached_logo_path = logo_path
        
        return logo_array
    
    def process_image_with_dynamic_chest_logo(self, img_path, logo_path, output_img_folder, i, 
                                        logo_size=0.15, crop_chest_bottom=True, 
                                        crop_bottom_percentage=0.5, crop_top_percentage=0.1):
        """Process image with dynamic chest logo positioning and proper occlusion handling + POST-PROCESSING"""
        start_time = time.time()
        
        img_cv2 = cv2.imread(str(img_path))
        
        # Load logo
        logo_image = self.load_and_validate_logo_cached(logo_path)
        if logo_image is None:
            print(f"Could not load logo image: {logo_path}")
            return
        
        fname, img_ext = os.path.splitext(os.path.basename(img_path))
        
        # Human detection
        det_out = self.detector(img_cv2)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        
        if not valid_idx.any():
            print(f"No humans detected in {img_path}")
            return
        
        # Process detected humans
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        bbox_scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0 
        bbox_center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0

        # Get camera intrinsics
        cam_int = self.get_cam_intrinsics(img_cv2)
        dataset = Dataset(img_cv2, bbox_center, bbox_scale, cam_int, False, img_path)
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

        # Collect all mesh data for proper occlusion handling
        all_mesh_data = []
        person_count = 0
        
        # Process batches
        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            img_h, img_w = batch['img_size'][0]
            
            with torch.no_grad():
                out_smpl_params, out_cam, focal_length_ = self.model(batch)

            # Get mesh output including joints
            output_vertices, output_joints, output_cam_trans = self.get_output_mesh(out_smpl_params, out_cam, batch)
            
            # Process each person and collect mesh data
            for person_idx in range(output_vertices.shape[0]):
                print(f"\nPreparing mesh data for person {person_count + 1}")
                
                vertices_np = output_vertices[person_idx].cpu().numpy()
                joints_3d = output_joints[person_idx].cpu().numpy()
                cam_trans = output_cam_trans[person_idx].cpu().numpy()
                
                # Create texture with dynamic chest calculation (NOW WITH MAGENTA BASE)
                texture_image, uv_coords, logo_bounds = self.chest_logo_3d.create_chest_focused_texture_dynamic(
                    vertices_np, self.smpl_model.faces, joints_3d, logo_image, logo_size
                )
                
                # Store mesh data for occlusion-aware rendering
                mesh_data = (
                    vertices_np,           # vertices
                    self.smpl_model.faces, # faces
                    texture_image,         # texture
                    uv_coords,            # uv coordinates
                    cam_trans,            # camera translation
                    person_count          # person ID
                )
                all_mesh_data.append(mesh_data)
                person_count += 1
        
        if not all_mesh_data:
            print("No valid mesh data generated")
            return
        
        print(f"\nRendering {len(all_mesh_data)} people with occlusion handling...")
        
        # Create occlusion-aware overlay
        overlay_image, rendered_mesh, debug_info = self.visualizer.create_occlusion_aware_overlay(
            img_cv2, all_mesh_data, cam_int, blend_alpha=0.8,
            crop_chest_bottom=crop_chest_bottom,
            crop_bottom_percentage=crop_bottom_percentage,
            crop_top_percentage=crop_top_percentage
        )
        
        # NEW: POST-PROCESSING TO REMOVE MAGENTA AND CREATE TRANSPARENCY
        print("\nApplying post-processing to remove magenta base and create transparency...")
        
        transparent_overlay, clean_overlay = self.visualizer.post_processor.remove_magenta_create_transparent(
            overlay_image, rendered_mesh, img_cv2
        )
        
        # Save results
        base_name = f"{fname}_{i:06d}"
        
        output_folder = output_img_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Save main results
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_original.jpg"), img_cv2)
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_occlusion_aware_overlay.jpg"), overlay_image)
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_rendered_mesh.png"), rendered_mesh)
        
        # NEW: Save post-processed results
        transparent_path, clean_path = self.visualizer.post_processor.save_transparent_results(
            transparent_overlay, clean_overlay, output_folder, base_name
        )
        
        # Save debug information
        if debug_info['depth_map'] is not None:
            # Normalize depth map for visualization
            depth_normalized = ((debug_info['depth_map'] - np.min(debug_info['depth_map'])) / 
                              (np.max(debug_info['depth_map']) - np.min(debug_info['depth_map']) + 1e-8) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_depth_map.png"), depth_normalized)
        
        # Save person ID map
        if debug_info['person_id_map'] is not None:
            person_id_normalized = ((debug_info['person_id_map'] + 1) * 50).astype(np.uint8)  # +1 to handle -1 values
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_person_id_map.png"), person_id_normalized)
        
        # Save individual person data
        for person_idx, mesh_data in enumerate(all_mesh_data):
            vertices_np, faces, texture_image, uv_coords, cam_trans, person_id = mesh_data
            
            person_base_name = f"{base_name}_person_{person_idx}"
            
            # Save individual texture (show magenta base for debugging)
            cv2.imwrite(os.path.join(output_folder, f"{person_base_name}_texture_with_magenta.png"), 
                       cv2.cvtColor(texture_image, cv2.COLOR_RGB2BGR))
            
            # Save individual mesh
            mesh = trimesh.Trimesh(vertices_np, faces, process=False)
            mesh.visual = trimesh.visual.TextureVisuals(
                uv=uv_coords,
                image=Image.fromarray(texture_image)
            )
            mesh.export(os.path.join(output_folder, f"{person_base_name}_mesh_with_logo.obj"))
                
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"\nImage processed in {processing_time:.2f} seconds ({person_count} persons)")
        print(f"Occlusion-aware rendering with post-processing completed successfully!")
        print(f"✓ Magenta base color removed")
        print(f"✓ Transparent overlay created")
        print(f"✓ Clean overlay without artifacts generated")
    
    def run_dynamic_chest_logo_pipeline(self, image_folder, logo_path, out_folder, logo_size=0.15, 
                                   crop_chest_bottom=True, crop_bottom_percentage=0.5, crop_top_percentage=0.1):
        """Run pipeline with dynamic chest logo positioning and occlusion handling + POST-PROCESSING"""
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        
        from glob import glob
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']
        images_list = [image for ext in image_extensions for image in glob(os.path.join(image_folder, ext))]
        
        print(f"=== ENHANCED DYNAMIC CHEST LOGO PIPELINE WITH POST-PROCESSING ===")
        print(f"Using logo: {logo_path}")
        print(f"Logo size: {logo_size}")
        print(f"Output folder: {out_folder}")
        print(f"Found {len(images_list)} images to process")
        print(f"Features:")
        print(f"  ✓ Occlusion handling: ENABLED")
        print(f"  ✓ Magenta base removal: ENABLED") 
        print(f"  ✓ Transparency creation: ENABLED")
        print(f"  ✓ Clean overlay generation: ENABLED")
        
        total_start_time = time.time()
        
        # Pre-load logo
        self.load_and_validate_logo_cached(logo_path)
        
        for ind, img_path in enumerate(images_list):
            print(f"\n{'='*60}")
            print(f"Processing image {ind+1}/{len(images_list)}: {img_path}")
            print(f"{'='*60}")
            
            self.process_image_with_dynamic_chest_logo(
                img_path, logo_path, out_folder, ind, logo_size=logo_size,
                crop_chest_bottom=crop_chest_bottom, 
                crop_bottom_percentage=crop_bottom_percentage, 
                crop_top_percentage=crop_top_percentage
            )
        
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        avg_time_per_image = total_time / len(images_list) if images_list else 0
        
        print(f"\n{'='*60}")
        print("ENHANCED DYNAMIC CHEST LOGO PIPELINE COMPLETED!")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average time per image: {avg_time_per_image:.2f} seconds")
        print(f"All results saved to: {out_folder}")
        print(f"✓ Proper occlusion handling between people")
        print(f"✓ Depth-aware logo rendering")
        print(f"✓ Magenta base color removal")
        print(f"✓ Transparent PNG outputs")
        print(f"✓ Clean overlay JPG outputs")
        print(f"✓ Individual person mesh exports")
        print(f"{'='*60}")

# Usage example
if __name__ == "__main__":
    import argparse
    
    def make_parser():
        parser = argparse.ArgumentParser(description='Enhanced Dynamic CameraHMR with Post-Processing Transparency')
        parser.add_argument("--image_folder", type=str, required=True, help="Path to input image folder.")
        parser.add_argument("--logo_path", type=str, required=True, help="Path to logo image.")
        parser.add_argument("--output_folder", type=str, required=True, help="Path to output folder.")
        parser.add_argument("--logo_size", type=float, default=0.3, help="Logo size on mesh (0.1 = small, 0.3 = large)")
        parser.add_argument("--crop_bottom_percentage", type=float, default=0.5, help="Percentage to crop from bottom (0.5 = 50%)")
        parser.add_argument("--crop_top_percentage", type=float, default=0.1, help="Percentage to crop from top (0.1 = 10%)")
        return parser

    parser = make_parser()
    args = parser.parse_args()
    
    # Use enhanced estimator with post-processing
    estimator = DynamicChestLogoHumanMeshEstimator()
    estimator.run_dynamic_chest_logo_pipeline(
        args.image_folder, 
        args.logo_path, 
        args.output_folder, 
        args.logo_size,
        crop_bottom_percentage=args.crop_bottom_percentage,
        crop_top_percentage=args.crop_top_percentage
    )