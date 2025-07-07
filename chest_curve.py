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

class ImprovedChestLogo3D:
    def __init__(self, smpl_model):
        self.smpl_model = smpl_model
        self.faces = smpl_model.faces
        self.chest_vertex_indices = self._get_anatomically_correct_chest_vertices()
        
    def _get_anatomically_correct_chest_vertices(self):
        """Get anatomically correct chest area vertex indices for SMPL model"""
        # IMPROVED: More precise chest vertices based on SMPL anatomy
        # These are the actual front chest vertices in SMPL topology
        chest_vertices = []
        
        # Primary chest region (sternum and pectoral area)
        # These are the most important vertices for logo placement
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
        
        return chest_vertices
    
    def validate_chest_vertices(self, vertices):
        """Enhanced validation for chest vertices"""
        valid_indices = [idx for idx in self.chest_vertex_indices if idx < len(vertices)]
        
        if len(valid_indices) < 20:  # Need more vertices for proper curvature
            print(f"Warning: Only {len(valid_indices)} valid chest vertices found")
            return False
        
        chest_vertices = vertices[valid_indices]
        
        # Check if vertices are on the front of the body
        avg_z = np.mean(chest_vertices[:, 2])
        if avg_z < 0:
            print("Warning: Chest vertices appear to be on the back")
            return False
        
        # Check for reasonable chest dimensions
        chest_width = np.max(chest_vertices[:, 0]) - np.min(chest_vertices[:, 0])
        chest_height = np.max(chest_vertices[:, 1]) - np.min(chest_vertices[:, 1])
        
        if chest_width < 0.05 or chest_height < 0.05:
            print(f"Warning: Chest region too small - width: {chest_width:.3f}, height: {chest_height:.3f}")
            return False
        
        return True
    
    def get_chest_region_info(self, vertices):
        """IMPROVED: Get chest region info with better curvature analysis"""
        valid_indices = [idx for idx in self.chest_vertex_indices if idx < len(vertices)]
        
        if not valid_indices or not self.validate_chest_vertices(vertices):
            return self._get_fallback_chest_info(vertices)
        
        chest_vertices = vertices[valid_indices]
        
        # IMPROVED: Use central chest vertices for better logo positioning
        # Focus on the sternum/central chest area
        central_indices = valid_indices[:30]  # First 30 are primary chest vertices
        central_chest_vertices = vertices[central_indices]
        
        # Calculate chest center from central vertices
        chest_center = np.mean(central_chest_vertices, axis=0)
        
        # IMPROVED: Better normal calculation using local surface fitting
        chest_normal = self._calculate_surface_normal(central_chest_vertices, chest_center)
        
        # Ensure normal points outward (front of body)
        if chest_normal[2] < 0:
            chest_normal = -chest_normal
        
        # Create orthonormal coordinate system
        world_up = np.array([0.0, 1.0, 0.0])
        
        # Calculate right vector (person's right, negative X in SMPL)
        right_vector = np.cross(world_up, chest_normal)
        right_vector = right_vector / (np.linalg.norm(right_vector) + 1e-8)
        
        # Recalculate up vector for orthogonality
        up_vector = np.cross(chest_normal, right_vector)
        up_vector = up_vector / (np.linalg.norm(up_vector) + 1e-8)
        
        # Ensure proper orientation
        if right_vector[0] > 0:
            right_vector = -right_vector
            up_vector = np.cross(chest_normal, right_vector)
            up_vector = up_vector / (np.linalg.norm(up_vector) + 1e-8)
        
        return chest_center, chest_normal, right_vector, up_vector
    
    def _calculate_surface_normal(self, surface_vertices, center_point):
        """Calculate surface normal using robust local surface fitting"""
        # Center the vertices
        centered_vertices = surface_vertices - center_point
        
        # Use PCA to find the principal directions
        cov_matrix = np.cov(centered_vertices.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # The eigenvector with smallest eigenvalue is the normal
        normal = eigenvectors[:, 0]  # First column (smallest eigenvalue)
        
        # Alternative: use cross product of two main directions
        if len(surface_vertices) >= 3:
            # Find two vectors in the surface
            v1 = surface_vertices[1] - surface_vertices[0]
            v2 = surface_vertices[2] - surface_vertices[0]
            
            # Cross product gives normal
            cross_normal = np.cross(v1, v2)
            cross_normal = cross_normal / (np.linalg.norm(cross_normal) + 1e-8)
            
            # Use the more reliable normal (dot product check)
            if np.dot(normal, cross_normal) < 0:
                normal = -normal
        
        return normal / (np.linalg.norm(normal) + 1e-8)
    
    def _get_fallback_chest_info(self, vertices):
        """Enhanced fallback chest positioning"""
        body_center = np.mean(vertices, axis=0)
        
        # Better chest positioning based on body proportions
        # Chest is typically 15-20% of body height below the head
        y_range = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
        chest_offset_y = -0.15 * y_range  # Move down from center
        
        chest_center = body_center + np.array([0.0, chest_offset_y, 0.08])
        chest_normal = np.array([0.0, 0.0, 1.0])
        right_vector = np.array([-1.0, 0.0, 0.0])  # Person's right
        up_vector = np.array([0.0, 1.0, 0.0])
        
        return chest_center, chest_normal, right_vector, up_vector
    
    def create_chest_focused_texture(self, vertices, faces, chest_center, right_vector, up_vector, logo_image, logo_size=0.15):
        """IMPROVED: Create texture with better curvature-aware UV mapping"""
        # Create base skin texture
        texture_size = 1024
        base_texture = np.full((texture_size, texture_size, 3), [210, 180, 140], dtype=np.uint8)
        
        # Resize logo maintaining aspect ratio
        logo_h, logo_w = logo_image.shape[:2]
        logo_aspect = logo_w / logo_h
        
        logo_pixel_size = int(texture_size * logo_size * 0.6)  # Adjusted for better sizing
        
        if logo_aspect > 1:
            logo_width = logo_pixel_size
            logo_height = int(logo_pixel_size / logo_aspect)
        else:
            logo_height = logo_pixel_size
            logo_width = int(logo_pixel_size * logo_aspect)
        
        logo_width = max(48, logo_width)
        logo_height = max(48, logo_height)
        
        logo_resized = cv2.resize(logo_image, (logo_width, logo_height), interpolation=cv2.INTER_LANCZOS4)
        
        # IMPROVED: Enhanced UV mapping with curvature following
        uv_coords = self._create_curvature_aware_uv_mapping(vertices, faces, chest_center, right_vector, up_vector)
        
        # IMPROVED: Better chest center positioning in UV space
        chest_u, chest_v = 0.5, 0.55  # Center horizontally, slightly higher vertically
        
        start_x = int((chest_u - 0.5 * logo_width / texture_size) * texture_size)
        start_y = int((chest_v - 0.5 * logo_height / texture_size) * texture_size)
        
        # Ensure logo fits within texture bounds
        start_x = max(0, min(start_x, texture_size - logo_width))
        start_y = max(0, min(start_y, texture_size - logo_height))
        
        # Apply logo with alpha blending
        if logo_resized.shape[2] == 4:
            alpha = logo_resized[:, :, 3:4] / 255.0
            logo_rgb = logo_resized[:, :, :3]
        else:
            alpha = np.ones((logo_height, logo_width, 1))
            logo_rgb = logo_resized
        
        # Apply logo to texture
        texture_region = base_texture[start_y:start_y+logo_height, start_x:start_x+logo_width]
        blended = (alpha * logo_rgb + (1 - alpha) * texture_region).astype(np.uint8)
        base_texture[start_y:start_y+logo_height, start_x:start_x+logo_width] = blended
        
        return base_texture, uv_coords, (start_x, start_y, logo_width, logo_height)
    
    def _create_curvature_aware_uv_mapping(self, vertices, faces, chest_center, right_vector, up_vector):
        """IMPROVED: Create UV mapping that follows body curvature more accurately"""
        # Get valid chest vertices
        valid_chest_indices = [idx for idx in self.chest_vertex_indices if idx < len(vertices)]
        
        if not valid_chest_indices:
            return self._create_fallback_uv_mapping(vertices, chest_center, right_vector, up_vector)
        
        # IMPROVED: Use distance-based weight mapping for curvature
        chest_vertices = vertices[valid_chest_indices]
        
        # Calculate distances from chest center to all vertices
        distances_to_chest = np.linalg.norm(vertices - chest_center, axis=1)
        
        # Create weight map based on proximity to chest
        max_chest_distance = np.max(np.linalg.norm(chest_vertices - chest_center, axis=1))
        chest_weights = np.exp(-distances_to_chest / (max_chest_distance * 0.5))
        
        # Transform vertices to chest coordinate system
        centered_vertices = vertices - chest_center
        
        # Project onto chest coordinate system
        u_coords = np.dot(centered_vertices, right_vector)
        v_coords = np.dot(centered_vertices, up_vector)
        depth_coords = np.dot(centered_vertices, up_vector)  # Distance from chest plane
        
        # IMPROVED: Apply curvature correction based on surface distance
        # Vertices further from the chest plane get UV coordinates adjusted
        curvature_correction = 1.0 + 0.2 * np.abs(depth_coords) / (max_chest_distance + 1e-8)
        
        # Calculate bounds for normalization
        chest_u = np.dot(chest_vertices - chest_center, right_vector)
        chest_v = np.dot(chest_vertices - chest_center, up_vector)
        
        u_range = max(np.max(chest_u) - np.min(chest_u), 0.1)
        v_range = max(np.max(chest_v) - np.min(chest_v), 0.1)
        
        u_center = np.mean(chest_u)
        v_center = np.mean(chest_v)
        
        # IMPROVED: Normalize with curvature and weight consideration
        u_normalized = (u_coords - u_center) / (u_range * 2 * curvature_correction) + 0.5
        v_normalized = (v_coords - v_center) / (v_range * 2 * curvature_correction) + 0.5
        
        # Apply chest-focused weighting
        # Vertices closer to chest get UV coordinates closer to chest center (0.5, 0.4)
        target_u, target_v = 0.5, 0.55
        u_normalized = chest_weights * target_u + (1 - chest_weights) * u_normalized
        v_normalized = chest_weights * target_v + (1 - chest_weights) * v_normalized
        
        # IMPROVED: Smooth UV coordinates using mesh topology
        u_normalized = self._smooth_uv_with_topology(u_normalized, faces, vertices, chest_center, 0.1)
        v_normalized = self._smooth_uv_with_topology(v_normalized, faces, vertices, chest_center, 0.1)
        
        # Clamp to valid range
        u_normalized = np.clip(u_normalized, 0.0, 1.0)
        v_normalized = np.clip(v_normalized, 0.0, 1.0)
        
        return np.column_stack([u_normalized, v_normalized])
    
    def _smooth_uv_with_topology(self, uv_component, faces, vertices, chest_center, smoothing_strength):
        """IMPROVED: Smooth UV coordinates considering mesh topology and distance from chest"""
        # Build vertex adjacency with weights
        vertex_adj = {}
        vertex_weights = {}
        
        distances_to_chest = np.linalg.norm(vertices - chest_center, axis=1)
        max_distance = np.max(distances_to_chest)
        
        for face in faces:
            for i in range(3):
                v1, v2, v3 = face[i], face[(i+1)%3], face[(i+2)%3]
                
                if v1 not in vertex_adj:
                    vertex_adj[v1] = []
                    vertex_weights[v1] = []
                
                # Calculate weights based on distance to chest
                weight2 = 1.0 / (1.0 + distances_to_chest[v2] / max_distance)
                weight3 = 1.0 / (1.0 + distances_to_chest[v3] / max_distance)
                
                vertex_adj[v1].extend([v2, v3])
                vertex_weights[v1].extend([weight2, weight3])
        
        # Apply weighted smoothing
        smoothed_uv = uv_component.copy()
        
        for vertex_idx in range(len(uv_component)):
            if vertex_idx in vertex_adj and vertex_adj[vertex_idx]:
                neighbors = vertex_adj[vertex_idx]
                weights = vertex_weights[vertex_idx]
                
                if neighbors and weights:
                    # Normalize weights
                    weights = np.array(weights)
                    weights = weights / (np.sum(weights) + 1e-8)
                    
                    # Weighted average of neighbors
                    neighbor_values = uv_component[neighbors]
                    weighted_avg = np.sum(neighbor_values * weights)
                    
                    # Blend with original value
                    # Vertices closer to chest get less smoothing to preserve detail
                    chest_distance_factor = distances_to_chest[vertex_idx] / max_distance
                    local_smoothing = smoothing_strength * chest_distance_factor
                    
                    smoothed_uv[vertex_idx] = (1 - local_smoothing) * uv_component[vertex_idx] + local_smoothing * weighted_avg
        
        return smoothed_uv
    
    def _create_fallback_uv_mapping(self, vertices, chest_center, right_vector, up_vector):
        """Enhanced fallback UV mapping"""
        centered_vertices = vertices - chest_center
        
        u_coords = np.dot(centered_vertices, right_vector)
        v_coords = np.dot(centered_vertices, up_vector)
        
        # Better normalization
        u_range = max(np.max(u_coords) - np.min(u_coords), 0.1)
        v_range = max(np.max(v_coords) - np.min(v_coords), 0.1)
        
        u_center = np.mean(u_coords)
        v_center = np.mean(v_coords)
        
        u_normalized = (u_coords - u_center) / (u_range * 2.5) + 0.5
        v_normalized = (v_coords - v_center) / (v_range * 2.5) + 0.4
        
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
        print("Directory created suc")
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
        parser.add_argument("--logo_size", type=float, default=0.5, help="Logo size on mesh (0.1 = small, 0.3 = large)")
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