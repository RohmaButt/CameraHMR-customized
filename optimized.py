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

class OptimizedChestLogo3D:
    def __init__(self, smpl_model):
        self.smpl_model = smpl_model
        self.faces = smpl_model.faces
        self.chest_vertex_indices = self._get_anatomically_correct_chest_vertices()
        
        # Cache frequently used computations
        self._cached_chest_indices = None
        self._cached_face_adjacency = None
        
    @lru_cache(maxsize=128)
    def _get_anatomically_correct_chest_vertices(self):
        """Get anatomically correct chest area vertex indices for SMPL model - CACHED"""
        # IMPROVED: More precise chest vertices based on SMPL anatomy
        # These are the actual front chest vertices in SMPL topology
        chest_vertices = []
        
        # Primary chest region (sternum and pectoral area)
        primary_chest = [
            1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209,
            1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219,
            1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229,
            1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239,
            1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249,
            1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259
        ]
        
        # Extended chest area for better curvature mapping
        extended_chest = [
            1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179,
            1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189,
            1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199,
            1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269,
            1270, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1279,
            1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1289
        ]
        
        # Upper chest (clavicle area)
        upper_chest = [
            1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149,
            1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159,
            1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169
        ]
        
        # Lower chest (below sternum)
        lower_chest = [
            1290, 1291, 1292, 1293, 1294, 1295, 1296, 1297, 1298, 1299,
            1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309,
            1310, 1311, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1319
        ]
        
        # Combine all chest vertices
        chest_vertices = primary_chest + extended_chest + upper_chest + lower_chest
        
        # Remove duplicates and sort
        chest_vertices = sorted(list(set(chest_vertices)))
        
        return tuple(chest_vertices)  # Return tuple for caching
    
    def validate_chest_vertices(self, vertices):
        """Enhanced validation for chest vertices - OPTIMIZED"""
        if self._cached_chest_indices is None:
            self._cached_chest_indices = [idx for idx in self.chest_vertex_indices if idx < len(vertices)]
        
        valid_indices = self._cached_chest_indices
        
        if len(valid_indices) < 20:
            return False
        
        # Vectorized operations
        chest_vertices = vertices[valid_indices]
        avg_z = np.mean(chest_vertices[:, 2])
        
        if avg_z < 0:
            return False
        
        # Vectorized min/max operations
        chest_ranges = np.ptp(chest_vertices, axis=0)  # Peak-to-peak (max - min)
        chest_width, chest_height = chest_ranges[0], chest_ranges[1]
        
        if chest_width < 0.05 or chest_height < 0.05:
            return False
        
        return True
    
    def get_chest_region_info(self, vertices):
        """OPTIMIZED: Get chest region info with vectorized computations"""
        if self._cached_chest_indices is None:
            self._cached_chest_indices = [idx for idx in self.chest_vertex_indices if idx < len(vertices)]
        
        valid_indices = self._cached_chest_indices
        
        if not valid_indices or not self.validate_chest_vertices(vertices):
            return self._get_fallback_chest_info(vertices)
        
        # Vectorized vertex selection
        chest_vertices = vertices[valid_indices]
        central_indices = valid_indices[:30]
        central_chest_vertices = vertices[central_indices]
        
        # Vectorized center calculation
        chest_center = np.mean(central_chest_vertices, axis=0)
        
        # Optimized normal calculation
        chest_normal = self._calculate_surface_normal_optimized(central_chest_vertices, chest_center)
        
        # Ensure normal points outward
        if chest_normal[2] < 0:
            chest_normal = -chest_normal
        
        # Vectorized coordinate system creation
        world_up = np.array([0.0, 1.0, 0.0])
        right_vector = np.cross(world_up, chest_normal)
        right_vector = right_vector / (np.linalg.norm(right_vector) + 1e-8)
        
        up_vector = np.cross(chest_normal, right_vector)
        up_vector = up_vector / (np.linalg.norm(up_vector) + 1e-8)
        
        # Ensure proper orientation
        if right_vector[0] > 0:
            right_vector = -right_vector
            up_vector = np.cross(chest_normal, right_vector)
            up_vector = up_vector / (np.linalg.norm(up_vector) + 1e-8)
        
        return chest_center, chest_normal, right_vector, up_vector
    
    def _calculate_surface_normal_optimized(self, surface_vertices, center_point):
        """OPTIMIZED: Calculate surface normal using vectorized operations"""
        # Vectorized centering
        centered_vertices = surface_vertices - center_point
        
        # Optimized covariance calculation
        cov_matrix = np.cov(centered_vertices.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        normal = eigenvectors[:, 0]
        
        # Vectorized cross product alternative
        if len(surface_vertices) >= 3:
            v1 = surface_vertices[1] - surface_vertices[0]
            v2 = surface_vertices[2] - surface_vertices[0]
            cross_normal = np.cross(v1, v2)
            cross_normal = cross_normal / (np.linalg.norm(cross_normal) + 1e-8)
            
            if np.dot(normal, cross_normal) < 0:
                normal = -normal
        
        return normal / (np.linalg.norm(normal) + 1e-8)
    
    def _get_fallback_chest_info(self, vertices):
        """OPTIMIZED: Enhanced fallback chest positioning with vectorized operations"""
        # Vectorized body center calculation
        body_center = np.mean(vertices, axis=0)
        
        # Vectorized range calculation
        y_range = np.ptp(vertices[:, 1])  # Peak-to-peak is faster than max - min
        chest_offset_y = -0.15 * y_range
        
        chest_center = body_center + np.array([0.0, chest_offset_y, 0.08])
        chest_normal = np.array([0.0, 0.0, 1.0])
        right_vector = np.array([-1.0, 0.0, 0.0])
        up_vector = np.array([0.0, 1.0, 0.0])
        
        return chest_center, chest_normal, right_vector, up_vector
    
    def create_chest_focused_texture(self, vertices, faces, chest_center, right_vector, up_vector, logo_image, logo_size=0.15):
        """OPTIMIZED: Create texture with parallel processing and vectorized operations"""
        # Pre-allocate arrays
        texture_size = 1024
        base_texture = np.full((texture_size, texture_size, 3), [210, 180, 140], dtype=np.uint8)
        
        # Optimized logo resizing
        logo_h, logo_w = logo_image.shape[:2]
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
        
        # Use faster interpolation for resizing
        logo_resized = cv2.resize(logo_image, (logo_width, logo_height), interpolation=cv2.INTER_LINEAR)
        
        # OPTIMIZED: Parallel UV mapping
        uv_coords = self._create_curvature_aware_uv_mapping_optimized(vertices, faces, chest_center, right_vector, up_vector)
        
        # Optimized logo placement
        chest_u, chest_v = 0.5, 0.55
        start_x = int((chest_u - 0.5 * logo_width / texture_size) * texture_size)
        start_y = int((chest_v - 0.5 * logo_height / texture_size) * texture_size)
        
        start_x = max(0, min(start_x, texture_size - logo_width))
        start_y = max(0, min(start_y, texture_size - logo_height))
        
        # Optimized alpha blending
        if logo_resized.shape[2] == 4:
            alpha = logo_resized[:, :, 3:4] / 255.0
            logo_rgb = logo_resized[:, :, :3]
        else:
            alpha = np.ones((logo_height, logo_width, 1))
            logo_rgb = logo_resized
        
        # Vectorized blending
        texture_region = base_texture[start_y:start_y+logo_height, start_x:start_x+logo_width]
        blended = (alpha * logo_rgb + (1 - alpha) * texture_region).astype(np.uint8)
        base_texture[start_y:start_y+logo_height, start_x:start_x+logo_width] = blended
        
        return base_texture, uv_coords, (start_x, start_y, logo_width, logo_height)
    
    def _create_curvature_aware_uv_mapping_optimized(self, vertices, faces, chest_center, right_vector, up_vector):
        """OPTIMIZED: Create UV mapping with vectorized operations and reduced smoothing"""
        if self._cached_chest_indices is None:
            self._cached_chest_indices = [idx for idx in self.chest_vertex_indices if idx < len(vertices)]
        
        valid_chest_indices = self._cached_chest_indices
        
        if not valid_chest_indices:
            return self._create_fallback_uv_mapping_optimized(vertices, chest_center, right_vector, up_vector)
        
        # Vectorized distance calculations
        chest_vertices = vertices[valid_chest_indices]
        distances_to_chest = np.linalg.norm(vertices - chest_center, axis=1)
        max_chest_distance = np.max(np.linalg.norm(chest_vertices - chest_center, axis=1))
        
        # Vectorized weight calculation
        chest_weights = np.exp(-distances_to_chest / (max_chest_distance * 0.5))
        
        # Vectorized transformations
        centered_vertices = vertices - chest_center
        u_coords = np.dot(centered_vertices, right_vector)
        v_coords = np.dot(centered_vertices, up_vector)
        depth_coords = np.dot(centered_vertices, up_vector)
        
        # Vectorized curvature correction
        curvature_correction = 1.0 + 0.2 * np.abs(depth_coords) / (max_chest_distance + 1e-8)
        
        # Vectorized bounds calculation
        chest_centered = chest_vertices - chest_center
        chest_u = np.dot(chest_centered, right_vector)
        chest_v = np.dot(chest_centered, up_vector)
        
        u_range = max(np.ptp(chest_u), 0.1)  # Peak-to-peak
        v_range = max(np.ptp(chest_v), 0.1)
        
        u_center = np.mean(chest_u)
        v_center = np.mean(chest_v)
        
        # Vectorized normalization
        u_normalized = (u_coords - u_center) / (u_range * 2 * curvature_correction) + 0.5
        v_normalized = (v_coords - v_center) / (v_range * 2 * curvature_correction) + 0.5
        
        # Vectorized weighting
        target_u, target_v = 0.5, 0.55
        u_normalized = chest_weights * target_u + (1 - chest_weights) * u_normalized
        v_normalized = chest_weights * target_v + (1 - chest_weights) * v_normalized
        
        # Reduced smoothing for speed
        u_normalized = self._smooth_uv_with_topology_optimized(u_normalized, faces, vertices, chest_center, 0.05)
        v_normalized = self._smooth_uv_with_topology_optimized(v_normalized, faces, vertices, chest_center, 0.05)
        
        # Vectorized clamping
        u_normalized = np.clip(u_normalized, 0.0, 1.0)
        v_normalized = np.clip(v_normalized, 0.0, 1.0)
        
        return np.column_stack([u_normalized, v_normalized])
    
    def _smooth_uv_with_topology_optimized(self, uv_component, faces, vertices, chest_center, smoothing_strength):
        """OPTIMIZED: Faster smoothing with reduced iterations and vectorized operations"""
        if self._cached_face_adjacency is None:
            self._cached_face_adjacency = self._build_face_adjacency_optimized(faces, len(vertices))
        
        vertex_adj, vertex_weights = self._cached_face_adjacency
        
        # Vectorized distance calculation
        distances_to_chest = np.linalg.norm(vertices - chest_center, axis=1)
        max_distance = np.max(distances_to_chest)
        
        smoothed_uv = uv_component.copy()
        
        # Optimized smoothing loop
        for vertex_idx in range(len(uv_component)):
            if vertex_idx in vertex_adj and vertex_adj[vertex_idx]:
                neighbors = vertex_adj[vertex_idx]
                weights = vertex_weights[vertex_idx]
                
                if neighbors and weights:
                    # Vectorized operations
                    weights = np.array(weights)
                    weights = weights / (np.sum(weights) + 1e-8)
                    neighbor_values = uv_component[neighbors]
                    weighted_avg = np.sum(neighbor_values * weights)
                    
                    # Reduced smoothing for speed
                    chest_distance_factor = distances_to_chest[vertex_idx] / max_distance
                    local_smoothing = smoothing_strength * chest_distance_factor * 0.5  # Reduced factor
                    
                    smoothed_uv[vertex_idx] = (1 - local_smoothing) * uv_component[vertex_idx] + local_smoothing * weighted_avg
        
        return smoothed_uv
    
    def _build_face_adjacency_optimized(self, faces, num_vertices):
        """OPTIMIZED: Build face adjacency with vectorized operations"""
        vertex_adj = {}
        vertex_weights = {}
        
        # Pre-allocate for known vertices
        for i in range(num_vertices):
            vertex_adj[i] = []
            vertex_weights[i] = []
        
        # Vectorized face processing
        for face in faces:
            for i in range(3):
                v1, v2, v3 = face[i], face[(i+1)%3], face[(i+2)%3]
                vertex_adj[v1].extend([v2, v3])
                vertex_weights[v1].extend([1.0, 1.0])  # Simplified weights for speed
        
        return vertex_adj, vertex_weights
    
    def _create_fallback_uv_mapping_optimized(self, vertices, chest_center, right_vector, up_vector):
        """OPTIMIZED: Enhanced fallback UV mapping with vectorized operations"""
        # Vectorized operations
        centered_vertices = vertices - chest_center
        u_coords = np.dot(centered_vertices, right_vector)
        v_coords = np.dot(centered_vertices, up_vector)
        
        # Vectorized normalization
        u_range = max(np.ptp(u_coords), 0.1)
        v_range = max(np.ptp(v_coords), 0.1)
        
        u_center = np.mean(u_coords)
        v_center = np.mean(v_coords)
        
        u_normalized = (u_coords - u_center) / (u_range * 2.5) + 0.5
        v_normalized = (v_coords - v_center) / (v_range * 2.5) + 0.4
        
        return np.column_stack([np.clip(u_normalized, 0.0, 1.0), np.clip(v_normalized, 0.0, 1.0)])

class OptimizedVisualization:
    """OPTIMIZED: Enhanced visualization with parallel processing and optimized rendering"""
    
    def __init__(self, width=1024, height=768):
        self.width = width
        self.height = height
        
    def render_textured_mesh_to_image_optimized(self, mesh_vertices, mesh_faces, texture_image, uv_coords, 
                                    camera_intrinsics, camera_translation, image_shape):
        """OPTIMIZED: Render textured 3D mesh with vectorized operations and parallel processing"""
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
        
        # Parallel triangle rasterization for large face counts
        if len(valid_faces) > 1000:
            self._rasterize_triangles_parallel(rendered_image, z_buffer, valid_faces, 
                                             vertices_2d, depths, uv_coords, texture_image)
        else:
            # Sequential for smaller face counts
            for face in valid_faces:
                triangle_2d = vertices_2d[face].astype(np.int32)
                triangle_depths = depths[face]
                triangle_uvs = uv_coords[face]
                
                self._rasterize_textured_triangle_optimized(
                    rendered_image, z_buffer, triangle_2d, triangle_depths, 
                    triangle_uvs, texture_image
                )
        
        return rendered_image
    
    def _rasterize_triangles_parallel(self, rendered_image, z_buffer, faces, vertices_2d, depths, uv_coords, texture):
        """OPTIMIZED: Parallel triangle rasterization"""
        # Split faces into chunks for parallel processing
        chunk_size = max(100, len(faces) // mp.cpu_count())
        face_chunks = [faces[i:i + chunk_size] for i in range(0, len(faces), chunk_size)]
        
        with ThreadPoolExecutor(max_workers=min(4, mp.cpu_count())) as executor:
            for chunk in face_chunks:
                executor.submit(self._rasterize_face_chunk, rendered_image, z_buffer, chunk, 
                              vertices_2d, depths, uv_coords, texture)
    
    def _rasterize_face_chunk(self, rendered_image, z_buffer, face_chunk, vertices_2d, depths, uv_coords, texture):
        """OPTIMIZED: Rasterize a chunk of faces"""
        for face in face_chunk:
            triangle_2d = vertices_2d[face].astype(np.int32)
            triangle_depths = depths[face]
            triangle_uvs = uv_coords[face]
            
            self._rasterize_textured_triangle_optimized(
                rendered_image, z_buffer, triangle_2d, triangle_depths, 
                triangle_uvs, texture
            )
    
    def _rasterize_textured_triangle_optimized(self, image, z_buffer, triangle_2d, depths, uvs, texture):
        """OPTIMIZED: Rasterize triangle with vectorized operations and early exit"""
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
        
        # Vectorized scanning with early exit optimizations
        y_range = np.arange(min_y, max_y + 1)
        x_range = np.arange(min_x, max_x + 1)
        
        # Pre-calculate some constants
        v1_y_minus_v2_y = v1[1] - v2[1]
        v2_x_minus_v1_x = v2[0] - v1[0]
        v2_y_minus_v0_y = v2[1] - v0[1]
        v0_x_minus_v2_x = v0[0] - v2[0]
        
        for y in y_range:
            # Pre-calculate y-dependent terms
            y_term1 = v1_y_minus_v2_y * (-v2[0]) + v2_x_minus_v1_x * (y - v2[1])
            y_term2 = v2_y_minus_v0_y * (-v2[0]) + v0_x_minus_v2_x * (y - v2[1])
            
            for x in x_range:
                # Optimized barycentric coordinate calculation
                w0 = (v1_y_minus_v2_y * x + y_term1) / denom
                w1 = (v2_y_minus_v0_y * x + y_term2) / denom
                w2 = 1 - w0 - w1
                
                # Early exit if outside triangle
                if w0 < 0 or w1 < 0 or w2 < 0:
                    continue
                
                # Interpolate depth
                depth = w0 * d0 + w1 * d1 + w2 * d2
                
                # Depth test with early exit
                if depth >= z_buffer[y, x]:
                    continue
                
                z_buffer[y, x] = depth
                
                # Optimized UV interpolation and texture sampling
                u = w0 * uv0[0] + w1 * uv1[0] + w2 * uv2[0]
                v = w0 * uv0[1] + w1 * uv1[1] + w2 * uv2[1]
                
                tex_x = int(np.clip(u * texture_w, 0, texture_w - 1))
                tex_y = int(np.clip(v * texture_h, 0, texture_h - 1))
                
                image[y, x] = texture[tex_y, tex_x]

    def create_textured_mesh_overlay_optimized(self, original_image, mesh_vertices, mesh_faces, texture_image, 
                               uv_coords, camera_intrinsics, camera_translation, blend_alpha=0.7, 
                               crop_chest_bottom=True, crop_percentage=0.5):
        """OPTIMIZED: Create overlay with vectorized operations"""
        # Use optimized rendering
        rendered_mesh = self.render_textured_mesh_to_image_optimized(
            mesh_vertices, mesh_faces, texture_image, uv_coords,
            camera_intrinsics, camera_translation, original_image.shape
        )
        
        # Convert with optimized operations
        rendered_mesh_bgr = cv2.cvtColor(rendered_mesh, cv2.COLOR_RGB2BGR)
        
        # Vectorized mask creation
        mesh_mask = np.any(rendered_mesh_bgr > 0, axis=2)
        
        # Optimized chest cropping
        if crop_chest_bottom:
            mesh_mask = self._apply_chest_bottom_cropping_optimized(
                mesh_mask, mesh_vertices, camera_intrinsics, camera_translation, 
                original_image.shape, crop_percentage
            )
        
        # Vectorized blending
        overlay_image = original_image.copy()
        overlay_image[mesh_mask] = (
            blend_alpha * rendered_mesh_bgr[mesh_mask] + 
            (1 - blend_alpha) * original_image[mesh_mask]
        ).astype(np.uint8)
        
        return overlay_image, rendered_mesh_bgr

    def _apply_chest_bottom_cropping_optimized(self, mesh_mask, mesh_vertices, camera_intrinsics, 
                               camera_translation, image_shape, crop_percentage=0.5):
        """OPTIMIZED: Apply chest cropping with vectorized operations"""
        h, w = image_shape[:2]
        
        # Vectorized transformations
        vertices_cam = mesh_vertices + camera_translation
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
        
        depths = vertices_cam[:, 2]
        valid_depth_mask = depths > 0.01
        
        if not np.any(valid_depth_mask):
            return mesh_mask
        
        # Vectorized visible pixel detection
        visible_pixels = np.where(mesh_mask)
        
        if len(visible_pixels[0]) == 0:
            return mesh_mask
        
        # Vectorized bounding box calculation
        min_y, max_y = np.min(visible_pixels[0]), np.max(visible_pixels[0])
        min_x, max_x = np.min(visible_pixels[1]), np.max(visible_pixels[1])
        
        # Calculate crop line
        chest_height = max_y - min_y
        crop_line_y = max_y - int(chest_height * crop_percentage)
        
        # Vectorized mask modification
        modified_mask = mesh_mask.copy()
        modified_mask[crop_line_y:max_y + 1, min_x:max_x + 1] = False
        
        return modified_mask
    
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
    
    def save_all_visualizations_optimized(self, output_folder, base_name, original_image, mesh_with_texture, 
                          overlay_image, texture_image, rendered_mesh=None, logo_bounds=None, combined_overlay=None):
        """OPTIMIZED: Save visualizations with parallel I/O"""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Prepare all save operations
        save_operations = [
            (os.path.join(output_folder, f"{base_name}_original.jpg"), original_image),
            (os.path.join(output_folder, f"{base_name}_textured_overlay.jpg"), overlay_image),
            (os.path.join(output_folder, f"{base_name}_texture.png"), cv2.cvtColor(texture_image, cv2.COLOR_RGB2BGR))
        ]
        
        if rendered_mesh is not None:
            save_operations.append((os.path.join(output_folder, f"{base_name}_rendered_mesh.png"), rendered_mesh))
        
        if combined_overlay is not None:
            save_operations.append((os.path.join(output_folder, f"{base_name}_combined_all_persons.jpg"), combined_overlay))
        
        # Parallel image saving
        with ThreadPoolExecutor(max_workers=4) as executor:
            for path, image in save_operations:
                executor.submit(cv2.imwrite, path, image)
        
        # Save 3D mesh
        if mesh_with_texture:
            mesh_with_texture.export(os.path.join(output_folder, f"{base_name}_mesh_with_logo.obj"))
        
        # Create texture visualization
        texture_viz = self.create_texture_visualization(texture_image, logo_bounds)
        texture_viz.savefig(os.path.join(output_folder, f"{base_name}_texture_analysis.png"), 
                           dpi=150, bbox_inches='tight')
        plt.close(texture_viz)
        
        print(f"All visualizations saved to: {output_folder}")

class UltraOptimized3DChestLogoHumanMeshEstimator(HumanMeshEstimator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chest_logo_3d = OptimizedChestLogo3D(self.smpl_model)
        self.visualizer = OptimizedVisualization()
        
        # Pre-load and cache logo
        self._cached_logo = None
        self._cached_logo_path = None
    
    @lru_cache(maxsize=32)
    def load_and_validate_logo_cached(self, logo_path):
        """OPTIMIZED: Load and validate logo image with caching"""
        if self._cached_logo_path == logo_path and self._cached_logo is not None:
            return self._cached_logo
        
        try:
            logo_pil = Image.open(logo_path).convert("RGBA")
            logo_array = np.array(logo_pil)
            
            if logo_array.shape[0] < 32 or logo_array.shape[1] < 32:
                print(f"Warning: Logo is very small ({logo_array.shape[1]}x{logo_array.shape[0]})")
            
            # Optimized color handling
            if logo_array.shape[2] == 4:
                alpha = logo_array[:, :, 3:4] / 255.0
                rgb = logo_array[:, :, :3]
                white_bg = np.full_like(rgb, 255)
                logo_rgb = (alpha * rgb + (1 - alpha) * white_bg).astype(np.uint8)
            else:
                logo_rgb = logo_array[:, :, :3]
            
            # Cache the result
            self._cached_logo = logo_rgb
            self._cached_logo_path = logo_path
            
            print(f"Logo loaded and cached successfully: {logo_rgb.shape}")
            return logo_rgb
            
        except Exception as e:
            print(f"Error loading logo: {e}")
            return None
    
    def process_image_with_textured_3d_chest_logo_optimized(self, img_path, logo_path, output_img_folder, i, 
                                            logo_size=0.15, crop_chest_bottom=True, crop_percentage=0.5):
        """OPTIMIZED: Process image with parallel processing and caching"""
        start_time = time.time()
        
        img_cv2 = cv2.imread(str(img_path))
        
        # Use cached logo loading
        logo_image = self.load_and_validate_logo_cached(logo_path)
        if logo_image is None:
            print(f"Could not load logo image: {logo_path}")
            return
        
        fname, img_ext = os.path.splitext(os.path.basename(img_path))
        
        # Optimized human detection
        det_out = self.detector(img_cv2)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        
        if not valid_idx.any():
            print(f"No humans detected in {img_path}")
            return
        
        # Vectorized bounding box processing
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        bbox_scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0 
        bbox_center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0

        # Get Camera intrinsics
        cam_int = self.get_cam_intrinsics(img_cv2)
        dataset = Dataset(img_cv2, bbox_center, bbox_scale, cam_int, False, img_path)
        
        # Optimized dataloader with more workers
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

        person_count = 0
        combined_overlay = img_cv2.copy()
        
        # Process all batches with GPU optimization
        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            img_h, img_w = batch['img_size'][0]
            
            with torch.no_grad():
                # Use mixed precision for faster inference
                with torch.cuda.amp.autocast():
                    out_smpl_params, out_cam, focal_length_ = self.model(batch)

            output_vertices, output_joints, output_cam_trans = self.get_output_mesh(out_smpl_params, out_cam, batch)
            
            # Parallel person processing
            person_data = []
            for person_idx in range(output_vertices.shape[0]):
                vertices_np = output_vertices[person_idx].cpu().numpy()
                cam_trans = output_cam_trans[person_idx].cpu().numpy()
                person_data.append((vertices_np, cam_trans))
            
            # Process persons with parallel operations where possible
            for person_idx, (vertices_np, cam_trans) in enumerate(person_data):
                print(f"\nProcessing person {person_count + 1}")
                
                # Optimized chest region detection
                chest_center, chest_normal, right_vector, up_vector = self.chest_logo_3d.get_chest_region_info(vertices_np)
                
                # Optimized texture creation
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
                
                # Optimized overlay creation
                overlay_image, rendered_mesh = self.visualizer.create_textured_mesh_overlay_optimized(
                    img_cv2, vertices_np, self.smpl_model.faces, texture_image, uv_coords,
                    cam_int, cam_trans, blend_alpha=0.8, 
                    crop_chest_bottom=crop_chest_bottom, crop_percentage=crop_percentage
                )

                # Update combined overlay
                combined_overlay, _ = self.visualizer.create_textured_mesh_overlay_optimized(
                    combined_overlay, vertices_np, self.smpl_model.faces, texture_image, uv_coords,
                    cam_int, cam_trans, blend_alpha=0.8,
                    crop_chest_bottom=crop_chest_bottom, crop_percentage=crop_percentage
                )
                
                # Generate base name
                base_name = f"{fname}_{i:06d}_person_{person_count}"
                
                # Optimized saving
                self.visualizer.save_all_visualizations_optimized(
                    output_img_folder, base_name, img_cv2, mesh, overlay_image, 
                    texture_image, rendered_mesh, logo_bounds, combined_overlay
                )
                
                person_count += 1
                
        if person_count > 0:
            combined_base_name = f"{fname}_{i:06d}_all_persons_combined"
            cv2.imwrite(os.path.join(output_img_folder, f"{combined_base_name}.jpg"), combined_overlay)
            
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Image processed in {processing_time:.2f} seconds ({person_count} persons)")
        print(f"Average time per person: {processing_time/max(1, person_count):.2f} seconds")

    def run_textured_3d_chest_logo_pipeline_optimized(self, image_folder, logo_path, out_folder, logo_size=0.15, 
                                       crop_chest_bottom=True, crop_percentage=0.5):
        """OPTIMIZED: Run pipeline with parallel image processing"""
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        
        from glob import glob
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']
        images_list = [image for ext in image_extensions for image in glob(os.path.join(image_folder, ext))]
        
        print(f"=== OPTIMIZED TEXTURED 3D CHEST LOGO PIPELINE ===")
        print(f"Using logo: {logo_path}")
        print(f"Logo size: {logo_size}")
        print(f"Output folder: {out_folder}")
        print(f"Found {len(images_list)} images to process")
        
        total_start_time = time.time()
        
        # Pre-load logo for caching
        self.load_and_validate_logo_cached(logo_path)
        
        for ind, img_path in enumerate(images_list):
            print(f"\n{'='*60}")
            print(f"Processing image {ind+1}/{len(images_list)}: {img_path}")
            print(f"{'='*60}")
            
            self.process_image_with_textured_3d_chest_logo_optimized(
                img_path, logo_path, out_folder, ind, logo_size=logo_size,
                crop_chest_bottom=crop_chest_bottom, crop_percentage=crop_percentage
            )
        
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        avg_time_per_image = total_time / len(images_list) if images_list else 0
        
        print(f"\n{'='*60}")
        print("OPTIMIZED TEXTURED 3D CHEST LOGO PIPELINE COMPLETED!")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average time per image: {avg_time_per_image:.2f} seconds")
        print(f"All results saved to: {out_folder}")
        print(f"{'='*60}")

# Usage example with ultra-optimized processing
if __name__ == "__main__":
    import argparse
    from glob import glob
    
    def make_parser():
        parser = argparse.ArgumentParser(description='Ultra-Optimized CameraHMR with Textured 3D Chest Logo Overlay')
        parser.add_argument("--image_folder", type=str, required=True, help="Path to input image folder.")
        parser.add_argument("--logo_path", type=str, required=True, help="Path to logo image.")
        parser.add_argument("--output_folder", type=str, required=True, help="Path to output folder.")
        parser.add_argument("--logo_size", type=float, default=0.3, help="Logo size on mesh (0.1 = small, 0.3 = large)")
        parser.add_argument("--blend_alpha", type=float, default=0.8, help="Blending alpha for overlay (0.0 = original image, 1.0 = full mesh)")
        return parser

    parser = make_parser()
    args = parser.parse_args()
    
    # Use ultra-optimized estimator
    estimator = UltraOptimized3DChestLogoHumanMeshEstimator()
    estimator.run_textured_3d_chest_logo_pipeline_optimized(
        args.image_folder, 
        args.logo_path, 
        args.output_folder, 
        args.logo_size
    )