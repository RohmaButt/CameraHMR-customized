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
        """Create texture with dynamic chest positioning based on joints"""
        # Pre-allocate arrays
        texture_size = 1024
        base_texture = np.full((texture_size, texture_size, 3), [210, 180, 140], dtype=np.uint8)
        
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
        
        # Create UV mapping based on dynamic chest coordinates - FIXED: Pass chest_normal
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
        """Create UV mapping using dynamic chest coordinates - FIXED: Added chest_normal parameter"""
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
        depth_coords = np.dot(centered_vertices, chest_normal)  # NOW chest_normal is defined
        
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

# Visualization class remains the same as in your original code
class OptimizedVisualization:
    """Enhanced visualization with parallel processing and optimized rendering"""
    
    def __init__(self, width=1024, height=768):
        self.width = width
        self.height = height
        
    def render_textured_mesh_to_image_optimized(self, mesh_vertices, mesh_faces, texture_image, uv_coords, 
                                    camera_intrinsics, camera_translation, image_shape):
        """Render textured 3D mesh with vectorized operations"""
        h, w = image_shape[:2]
        rendered_image = np.zeros((h, w, 3), dtype=np.uint8)
        z_buffer = np.full((h, w), float('inf'))
        
        # Vectorized vertex transformation
        vertices_cam = mesh_vertices + camera_translation
        
        # Vectorized projection
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
        
        depths = vertices_cam[:, 2]
        valid_depth_mask = depths > 0.01
        
        # Vectorized perspective projection
        vertices_2d = np.zeros((len(vertices_cam), 2))
        valid_indices = np.where(valid_depth_mask)[0]
        
        if len(valid_indices) > 0:
            vertices_2d[valid_indices, 0] = (vertices_cam[valid_indices, 0] * fx / depths[valid_indices]) + cx
            vertices_2d[valid_indices, 1] = (vertices_cam[valid_indices, 1] * fy / depths[valid_indices]) + cy
        
        # Filter faces with all valid vertices
        valid_faces = []
        for face in mesh_faces:
            if all(valid_depth_mask[idx] for idx in face):
                triangle_2d = vertices_2d[face].astype(np.int32)
                if not (np.any(triangle_2d[:, 0] < 0) or np.any(triangle_2d[:, 0] >= w) or
                       np.any(triangle_2d[:, 1] < 0) or np.any(triangle_2d[:, 1] >= h)):
                    valid_faces.append(face)
        
        # Rasterize triangles
        for face in valid_faces:
            triangle_2d = vertices_2d[face].astype(np.int32)
            triangle_depths = depths[face]
            triangle_uvs = uv_coords[face]
            
            self._rasterize_textured_triangle_optimized(
                rendered_image, z_buffer, triangle_2d, triangle_depths, 
                triangle_uvs, texture_image
            )
        
        return rendered_image
    
    def _rasterize_textured_triangle_optimized(self, image, z_buffer, triangle_2d, depths, uvs, texture):
        """Rasterize triangle with vectorized operations"""
        # Optimized bounding box calculation
        min_x = max(0, np.min(triangle_2d[:, 0]))
        max_x = min(image.shape[1] - 1, np.max(triangle_2d[:, 0]))
        min_y = max(0, np.min(triangle_2d[:, 1]))
        max_y = min(image.shape[0] - 1, np.max(triangle_2d[:, 1]))
        
        if min_x >= max_x or min_y >= max_y:
            return
        
        v0, v1, v2 = triangle_2d
        uv0, uv1, uv2 = uvs
        d0, d1, d2 = depths
        
        # Precompute triangle area
        denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
        if abs(denom) < 1e-8:
            return
        
        texture_h, texture_w = texture.shape[:2]
        
        # Optimized scanning
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                # Barycentric coordinates
                w0 = ((v1[1] - v2[1]) * (x - v2[0]) + (v2[0] - v1[0]) * (y - v2[1])) / denom
                w1 = ((v2[1] - v0[1]) * (x - v2[0]) + (v0[0] - v2[0]) * (y - v2[1])) / denom
                w2 = 1 - w0 - w1
                
                # Check if inside triangle
                if w0 < 0 or w1 < 0 or w2 < 0:
                    continue
                
                # Interpolate depth
                depth = w0 * d0 + w1 * d1 + w2 * d2
                
                # Depth test
                if depth >= z_buffer[y, x]:
                    continue
                
                z_buffer[y, x] = depth
                
                # Interpolate UV coordinates
                u = w0 * uv0[0] + w1 * uv1[0] + w2 * uv2[0]
                v = w0 * uv0[1] + w1 * uv1[1] + w2 * uv2[1]
                
                # Sample texture
                tex_x = int(np.clip(u * texture_w, 0, texture_w - 1))
                tex_y = int(np.clip(v * texture_h, 0, texture_h - 1))
                
                image[y, x] = texture[tex_y, tex_x]

    def create_textured_mesh_overlay_optimized(self, original_image, mesh_vertices, mesh_faces, texture_image, 
                           uv_coords, camera_intrinsics, camera_translation, blend_alpha=0.7, 
                           crop_chest_bottom=True, crop_bottom_percentage=0.5, crop_top_percentage=0.1):
        """Create overlay with optimized rendering"""
        # Use optimized rendering
        rendered_mesh = self.render_textured_mesh_to_image_optimized(
            mesh_vertices, mesh_faces, texture_image, uv_coords,
            camera_intrinsics, camera_translation, original_image.shape
        )
        
        # Convert colors
        rendered_mesh_bgr = cv2.cvtColor(rendered_mesh, cv2.COLOR_RGB2BGR)
        
        # Create mask
        mesh_mask = np.any(rendered_mesh_bgr > 0, axis=2)
        
        # Apply cropping if needed
        if crop_chest_bottom:
            mesh_mask = self._apply_chest_cropping_optimized(
                mesh_mask, mesh_vertices, camera_intrinsics, camera_translation, 
                original_image.shape, crop_bottom_percentage, crop_top_percentage
            )
        
        # Blend images
        overlay_image = original_image.copy()
        overlay_image[mesh_mask] = (
            blend_alpha * rendered_mesh_bgr[mesh_mask] + 
            (1 - blend_alpha) * original_image[mesh_mask]
        ).astype(np.uint8)
        
        return overlay_image, rendered_mesh_bgr

    def _apply_chest_cropping_optimized(self, mesh_mask, mesh_vertices, camera_intrinsics, 
                           camera_translation, image_shape, crop_bottom_percentage=0.5, crop_top_percentage=0.1):
        """Apply chest cropping with optimized operations"""
        h, w = image_shape[:2]
        
        # Find visible pixels
        visible_pixels = np.where(mesh_mask)
        
        if len(visible_pixels[0]) == 0:
            return mesh_mask
        
        # Calculate bounding box
        min_y, max_y = np.min(visible_pixels[0]), np.max(visible_pixels[0])
        min_x, max_x = np.min(visible_pixels[1]), np.max(visible_pixels[1])
        
        # Calculate crop lines
        chest_height = max_y - min_y
        bottom_crop_line_y = max_y - int(chest_height * crop_bottom_percentage)
        top_crop_line_y = min_y + int(chest_height * crop_top_percentage)
        
        # Apply cropping
        modified_mask = mesh_mask.copy()
        modified_mask[bottom_crop_line_y:max_y + 1, min_x:max_x + 1] = False
        modified_mask[min_y:top_crop_line_y + 1, min_x:max_x + 1] = False
        
        return modified_mask
# Visualization class remains the same as in your original code
class OptimizedVisualization:
    """Enhanced visualization with parallel processing and optimized rendering"""
    
    def __init__(self, width=1024, height=768):
        self.width = width
        self.height = height
        
    def render_textured_mesh_to_image_optimized(self, mesh_vertices, mesh_faces, texture_image, uv_coords, 
                                    camera_intrinsics, camera_translation, image_shape):
        """Render textured 3D mesh with vectorized operations"""
        h, w = image_shape[:2]
        rendered_image = np.zeros((h, w, 3), dtype=np.uint8)
        z_buffer = np.full((h, w), float('inf'))
        
        # Vectorized vertex transformation
        vertices_cam = mesh_vertices + camera_translation
        
        # Vectorized projection
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
        
        depths = vertices_cam[:, 2]
        valid_depth_mask = depths > 0.01
        
        # Vectorized perspective projection
        vertices_2d = np.zeros((len(vertices_cam), 2))
        valid_indices = np.where(valid_depth_mask)[0]
        
        if len(valid_indices) > 0:
            vertices_2d[valid_indices, 0] = (vertices_cam[valid_indices, 0] * fx / depths[valid_indices]) + cx
            vertices_2d[valid_indices, 1] = (vertices_cam[valid_indices, 1] * fy / depths[valid_indices]) + cy
        
        # Filter faces with all valid vertices
        valid_faces = []
        for face in mesh_faces:
            if all(valid_depth_mask[idx] for idx in face):
                triangle_2d = vertices_2d[face].astype(np.int32)
                if not (np.any(triangle_2d[:, 0] < 0) or np.any(triangle_2d[:, 0] >= w) or
                       np.any(triangle_2d[:, 1] < 0) or np.any(triangle_2d[:, 1] >= h)):
                    valid_faces.append(face)
        
        # Rasterize triangles
        for face in valid_faces:
            triangle_2d = vertices_2d[face].astype(np.int32)
            triangle_depths = depths[face]
            triangle_uvs = uv_coords[face]
            
            self._rasterize_textured_triangle_optimized(
                rendered_image, z_buffer, triangle_2d, triangle_depths, 
                triangle_uvs, texture_image
            )
        
        return rendered_image
    
    def _rasterize_textured_triangle_optimized(self, image, z_buffer, triangle_2d, depths, uvs, texture):
        """Rasterize triangle with vectorized operations"""
        # Optimized bounding box calculation
        min_x = max(0, np.min(triangle_2d[:, 0]))
        max_x = min(image.shape[1] - 1, np.max(triangle_2d[:, 0]))
        min_y = max(0, np.min(triangle_2d[:, 1]))
        max_y = min(image.shape[0] - 1, np.max(triangle_2d[:, 1]))
        
        if min_x >= max_x or min_y >= max_y:
            return
        
        v0, v1, v2 = triangle_2d
        uv0, uv1, uv2 = uvs
        d0, d1, d2 = depths
        
        # Precompute triangle area
        denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
        if abs(denom) < 1e-8:
            return
        
        texture_h, texture_w = texture.shape[:2]
        
        # Optimized scanning
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                # Barycentric coordinates
                w0 = ((v1[1] - v2[1]) * (x - v2[0]) + (v2[0] - v1[0]) * (y - v2[1])) / denom
                w1 = ((v2[1] - v0[1]) * (x - v2[0]) + (v0[0] - v2[0]) * (y - v2[1])) / denom
                w2 = 1 - w0 - w1
                
                # Check if inside triangle
                if w0 < 0 or w1 < 0 or w2 < 0:
                    continue
                
                # Interpolate depth
                depth = w0 * d0 + w1 * d1 + w2 * d2
                
                # Depth test
                if depth >= z_buffer[y, x]:
                    continue
                
                z_buffer[y, x] = depth
                
                # Interpolate UV coordinates
                u = w0 * uv0[0] + w1 * uv1[0] + w2 * uv2[0]
                v = w0 * uv0[1] + w1 * uv1[1] + w2 * uv2[1]
                
                # Sample texture
                tex_x = int(np.clip(u * texture_w, 0, texture_w - 1))
                tex_y = int(np.clip(v * texture_h, 0, texture_h - 1))
                
                image[y, x] = texture[tex_y, tex_x]

    def create_textured_mesh_overlay_optimized(self, original_image, mesh_vertices, mesh_faces, texture_image, 
                           uv_coords, camera_intrinsics, camera_translation, blend_alpha=0.7, 
                           crop_chest_bottom=True, crop_bottom_percentage=0.5, crop_top_percentage=0.1):
        """Create overlay with optimized rendering"""
        # Use optimized rendering
        rendered_mesh = self.render_textured_mesh_to_image_optimized(
            mesh_vertices, mesh_faces, texture_image, uv_coords,
            camera_intrinsics, camera_translation, original_image.shape
        )
        
        # Convert colors
        rendered_mesh_bgr = cv2.cvtColor(rendered_mesh, cv2.COLOR_RGB2BGR)
        
        # Create mask
        mesh_mask = np.any(rendered_mesh_bgr > 0, axis=2)
        
        # Apply cropping if needed
        if crop_chest_bottom:
            mesh_mask = self._apply_chest_cropping_optimized(
                mesh_mask, mesh_vertices, camera_intrinsics, camera_translation, 
                original_image.shape, crop_bottom_percentage, crop_top_percentage
            )
        
        # Blend images
        overlay_image = original_image.copy()
        overlay_image[mesh_mask] = (
            blend_alpha * rendered_mesh_bgr[mesh_mask] + 
            (1 - blend_alpha) * original_image[mesh_mask]
        ).astype(np.uint8)
        
        return overlay_image, rendered_mesh_bgr

    def _apply_chest_cropping_optimized(self, mesh_mask, mesh_vertices, camera_intrinsics, 
                           camera_translation, image_shape, crop_bottom_percentage=0.5, crop_top_percentage=0.1):
        """Apply chest cropping with optimized operations"""
        h, w = image_shape[:2]
        
        # Find visible pixels
        visible_pixels = np.where(mesh_mask)
        
        if len(visible_pixels[0]) == 0:
            return mesh_mask
        
        # Calculate bounding box
        min_y, max_y = np.min(visible_pixels[0]), np.max(visible_pixels[0])
        min_x, max_x = np.min(visible_pixels[1]), np.max(visible_pixels[1])
        
        # Calculate crop lines
        chest_height = max_y - min_y
        bottom_crop_line_y = max_y - int(chest_height * crop_bottom_percentage)
        top_crop_line_y = min_y + int(chest_height * crop_top_percentage)
        
        # Apply cropping
        modified_mask = mesh_mask.copy()
        modified_mask[bottom_crop_line_y:max_y + 1, min_x:max_x + 1] = False
        modified_mask[min_y:top_crop_line_y + 1, min_x:max_x + 1] = False
        
        return modified_mask

# Updated main estimator class
class DynamicChestLogoHumanMeshEstimator(HumanMeshEstimator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chest_logo_3d = DynamicChestLogo3D(self.smpl_model)
        self.visualizer = OptimizedVisualization()
        
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
        """Process image with dynamic chest logo positioning"""
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

        person_count = 0
        combined_overlay = img_cv2.copy()
        
        # Process batches
        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            img_h, img_w = batch['img_size'][0]
            
            with torch.no_grad():
                out_smpl_params, out_cam, focal_length_ = self.model(batch)

            # Get mesh output including joints
            output_vertices, output_joints, output_cam_trans = self.get_output_mesh(out_smpl_params, out_cam, batch)
            
            # Process each person
            for person_idx in range(output_vertices.shape[0]):
                print(f"\nProcessing person {person_count + 1}")
                
                vertices_np = output_vertices[person_idx].cpu().numpy()
                joints_3d = output_joints[person_idx].cpu().numpy()  # This is the key addition!
                cam_trans = output_cam_trans[person_idx].cpu().numpy()
                
                # Use dynamic chest calculation with joints
                texture_image, uv_coords, logo_bounds = self.chest_logo_3d.create_chest_focused_texture_dynamic(
                    vertices_np, self.smpl_model.faces, joints_3d, logo_image, logo_size
                )
                
                # Create mesh with texture
                mesh = trimesh.Trimesh(vertices_np, self.smpl_model.faces, process=False)
                mesh.visual = trimesh.visual.TextureVisuals(
                    uv=uv_coords,
                    image=Image.fromarray(texture_image)
                )
                
                # Create overlay
                overlay_image, rendered_mesh = self.visualizer.create_textured_mesh_overlay_optimized(
                    img_cv2, vertices_np, self.smpl_model.faces, texture_image, uv_coords,
                    cam_int, cam_trans, blend_alpha=0.8, 
                    crop_chest_bottom=crop_chest_bottom, 
                    crop_bottom_percentage=crop_bottom_percentage, 
                    crop_top_percentage=crop_top_percentage
                )

                # Update combined overlay
                combined_overlay, _ = self.visualizer.create_textured_mesh_overlay_optimized(
                    combined_overlay, vertices_np, self.smpl_model.faces, texture_image, uv_coords,
                    cam_int, cam_trans, blend_alpha=0.8,
                    crop_chest_bottom=crop_chest_bottom, 
                    crop_bottom_percentage=crop_bottom_percentage, 
                    crop_top_percentage=crop_top_percentage
                )
                
                # Save results
                base_name = f"{fname}_{i:06d}_person_{person_count}"
                
                output_folder = output_img_folder
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                
                cv2.imwrite(os.path.join(output_folder, f"{base_name}_original.jpg"), img_cv2)
                cv2.imwrite(os.path.join(output_folder, f"{base_name}_textured_overlay.jpg"), overlay_image)
                cv2.imwrite(os.path.join(output_folder, f"{base_name}_texture.png"), cv2.cvtColor(texture_image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(output_folder, f"{base_name}_rendered_mesh.png"), rendered_mesh)
                
                mesh.export(os.path.join(output_folder, f"{base_name}_mesh_with_logo.obj"))
                
                person_count += 1
                
        if person_count > 0:
            combined_base_name = f"{fname}_{i:06d}_all_persons_combined"
            cv2.imwrite(os.path.join(output_img_folder, f"{combined_base_name}.jpg"), combined_overlay)
            
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Image processed in {processing_time:.2f} seconds ({person_count} persons)")
    
    def run_dynamic_chest_logo_pipeline(self, image_folder, logo_path, out_folder, logo_size=0.15, 
                                   crop_chest_bottom=True, crop_bottom_percentage=0.5, crop_top_percentage=0.1):
        """Run pipeline with dynamic chest logo positioning"""
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        
        from glob import glob
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']
        images_list = [image for ext in image_extensions for image in glob(os.path.join(image_folder, ext))]
        
        print(f"=== DYNAMIC CHEST LOGO PIPELINE ===")
        print(f"Using logo: {logo_path}")
        print(f"Logo size: {logo_size}")
        print(f"Output folder: {out_folder}")
        print(f"Found {len(images_list)} images to process")
        
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
        print("DYNAMIC CHEST LOGO PIPELINE COMPLETED!")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average time per image: {avg_time_per_image:.2f} seconds")
        print(f"All results saved to: {out_folder}")
        print(f"{'='*60}")

# Usage example
if __name__ == "__main__":
    import argparse
    
    def make_parser():
        parser = argparse.ArgumentParser(description='Dynamic CameraHMR with Joint-based Chest Logo Positioning')
        parser.add_argument("--image_folder", type=str, required=True, help="Path to input image folder.")
        parser.add_argument("--logo_path", type=str, required=True, help="Path to logo image.")
        parser.add_argument("--output_folder", type=str, required=True, help="Path to output folder.")
        parser.add_argument("--logo_size", type=float, default=0.3, help="Logo size on mesh (0.1 = small, 0.3 = large)")
        parser.add_argument("--crop_bottom_percentage", type=float, default=0.5, help="Percentage to crop from bottom (0.5 = 50%)")
        parser.add_argument("--crop_top_percentage", type=float, default=0.1, help="Percentage to crop from top (0.1 = 10%)")
        return parser

    parser = make_parser()
    args = parser.parse_args()
    
    # Use dynamic estimator
    estimator = DynamicChestLogoHumanMeshEstimator()
    estimator.run_dynamic_chest_logo_pipeline(
        args.image_folder, 
        args.logo_path, 
        args.output_folder, 
        args.logo_size,
        crop_bottom_percentage=args.crop_bottom_percentage,
        crop_top_percentage=args.crop_top_percentage
    )