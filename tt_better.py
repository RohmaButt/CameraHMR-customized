
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

class TextureTransferFromSample:
    def __init__(self, smpl_model):
        self.smpl_model = smpl_model
        self.faces = smpl_model.faces
        
    def create_body_part_masks(self, vertices):
        """Create masks for different body parts based on vertex positions"""
        # SMPL vertex indices for different body parts (approximate)
        body_parts = {
            'head': list(range(0, 100)),  # Head vertices
            'torso': list(range(100, 1000)),  # Torso vertices
            'left_arm': list(range(1000, 2000)),  # Left arm
            'right_arm': list(range(2000, 3000)),  # Right arm
            'left_leg': list(range(3000, 4500)),  # Left leg
            'right_leg': list(range(4500, 6000)),  # Right leg
            'hands': list(range(6000, 6890))  # Hands and feet
        }
        return body_parts
    
    def extract_texture_regions_from_sample(self, sample_img, body_parts):
        """Extract texture regions from sample image for different body parts"""
        h, w = sample_img.shape[:2]
        
        # Define regions in the sample image for different body parts
        regions = {
            'head': sample_img[0:h//4, w//4:3*w//4],  # Top center
            'torso': sample_img[h//4:3*h//4, w//4:3*w//4],  # Center
            'left_arm': sample_img[h//4:3*h//4, 0:w//4],  # Left side
            'right_arm': sample_img[h//4:3*h//4, 3*w//4:w],  # Right side
            'left_leg': sample_img[3*h//4:h, 0:w//2],  # Bottom left
            'right_leg': sample_img[3*h//4:h, w//2:w],  # Bottom right
            'hands': sample_img[0:h//6, 0:w//6]  # Top left corner
        }
        return regions
    
    def sample_texture_from_region(self, region, num_samples):
        """Sample colors from a texture region"""
        if region.size == 0:
            return np.array([[128, 128, 128]] * num_samples)  # Default gray
        
        h, w = region.shape[:2]
        colors = []
        
        for _ in range(num_samples):
            # Random sampling from the region
            y = np.random.randint(0, h)
            x = np.random.randint(0, w)
            color = region[y, x]
            colors.append(color)
        
        return np.array(colors)
    
    def apply_texture_to_vertices(self, vertices, sample_img, method='body_parts'):
        """Apply texture from sample image to vertices"""
        num_vertices = len(vertices)
        vertex_colors = np.zeros((num_vertices, 3), dtype=np.uint8)
        
        if method == 'body_parts':
            # Map different body parts to different regions of the sample image
            body_parts = self.create_body_part_masks(vertices)
            texture_regions = self.extract_texture_regions_from_sample(sample_img, body_parts)
            
            for part_name, vertex_indices in body_parts.items():
                if part_name in texture_regions:
                    region = texture_regions[part_name]
                    num_part_vertices = len(vertex_indices)
                    
                    if num_part_vertices > 0:
                        part_colors = self.sample_texture_from_region(region, num_part_vertices)
                        vertex_colors[vertex_indices] = part_colors
        
        elif method == 'spherical_mapping':
            # Map the entire sample image using spherical coordinates
            vertex_colors = self.apply_spherical_texture_mapping(vertices, sample_img)
        
        elif method == 'cylindrical_mapping':
            # Map using cylindrical projection
            vertex_colors = self.apply_cylindrical_texture_mapping(vertices, sample_img)
        
        return vertex_colors
    
    def apply_spherical_texture_mapping(self, vertices, sample_img):
        """Apply texture using spherical mapping"""
        h, w = sample_img.shape[:2]
        
        # Normalize vertices to unit sphere
        vertices_centered = vertices - np.mean(vertices, axis=0)
        vertices_norm = vertices_centered / (np.linalg.norm(vertices_centered, axis=1, keepdims=True) + 1e-8)
        
        # Convert to spherical coordinates
        theta = np.arctan2(vertices_norm[:, 2], vertices_norm[:, 0])  # azimuth [-π, π]
        phi = np.arcsin(np.clip(vertices_norm[:, 1], -1, 1))  # elevation [-π/2, π/2]
        
        # Map to texture coordinates
        u = (theta + np.pi) / (2 * np.pi)  # [0, 1]
        v = (phi + np.pi/2) / np.pi  # [0, 1]
        
        # Convert to pixel coordinates
        u_pix = np.clip((u * (w - 1)).astype(int), 0, w - 1)
        v_pix = np.clip((v * (h - 1)).astype(int), 0, h - 1)
        
        # Sample colors from texture
        vertex_colors = sample_img[v_pix, u_pix]
        
        return vertex_colors
    
    def apply_cylindrical_texture_mapping(self, vertices, sample_img):
        """Apply texture using cylindrical mapping"""
        h, w = sample_img.shape[:2]
        
        # Center vertices
        vertices_centered = vertices - np.mean(vertices, axis=0)
        
        # Cylindrical coordinates
        theta = np.arctan2(vertices_centered[:, 2], vertices_centered[:, 0])  # azimuth
        y_coord = vertices_centered[:, 1]  # height
        
        # Map to texture coordinates
        u = (theta + np.pi) / (2 * np.pi)  # [0, 1]
        v = (y_coord - np.min(y_coord)) / (np.max(y_coord) - np.min(y_coord) + 1e-8)  # [0, 1]
        
        # Convert to pixel coordinates
        u_pix = np.clip((u * (w - 1)).astype(int), 0, w - 1)
        v_pix = np.clip((v * (h - 1)).astype(int), 0, h - 1)
        
        # Sample colors from texture
        vertex_colors = sample_img[v_pix, u_pix]
        
        return vertex_colors
    
    def smooth_vertex_colors(self, colors, vertices, k=5):
        """Smooth vertex colors using k-nearest neighbors"""
        if len(vertices) < k:
            return colors
            
        # Find k-nearest neighbors for each vertex
        nbrs = NearestNeighbors(n_neighbors=min(k+1, len(vertices)), algorithm='ball_tree').fit(vertices)
        distances, indices = nbrs.kneighbors(vertices)
        
        smoothed_colors = np.copy(colors)
        
        for i in range(len(vertices)):
            # Get neighbor colors (excluding self)
            neighbor_indices = indices[i][1:]  # Skip first (self)
            if len(neighbor_indices) > 0:
                neighbor_colors = colors[neighbor_indices]
                neighbor_distances = distances[i][1:]
                
                # Weight by inverse distance
                weights = 1.0 / (neighbor_distances + 1e-8)
                weights = weights / np.sum(weights)
                
                # Weighted average of neighbor colors
                smoothed_color = np.average(neighbor_colors, axis=0, weights=weights)
                smoothed_colors[i] = 0.7 * colors[i] + 0.3 * smoothed_color
        
        return smoothed_colors.astype(np.uint8)
    
    def create_textured_mesh(self, vertices, colors, output_path):
        """Create a mesh with vertex colors and save as PLY file"""
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=self.faces,
            vertex_colors=colors,
            process=False
        )
        
        # Save as PLY to preserve vertex colors
        ply_path = output_path.replace('.obj', '.ply')
        mesh.export(ply_path)
        
        return mesh, ply_path
    
    def visualize_texture_mapping(self, sample_img, method='body_parts'):
        """Visualize how the sample image will be mapped"""
        h, w = sample_img.shape[:2]
        
        if method == 'body_parts':
            # Show different regions
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.flatten()
            
            regions = {
                'head': sample_img[0:h//4, w//4:3*w//4],
                'torso': sample_img[h//4:3*h//4, w//4:3*w//4],
                'left_arm': sample_img[h//4:3*h//4, 0:w//4],
                'right_arm': sample_img[h//4:3*h//4, 3*w//4:w],
                'left_leg': sample_img[3*h//4:h, 0:w//2],
                'right_leg': sample_img[3*h//4:h, w//2:w],
                'hands': sample_img[0:h//6, 0:w//6]
            }
            
            for i, (part_name, region) in enumerate(regions.items()):
                if i < len(axes):
                    axes[i].imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
                    axes[i].set_title(f'{part_name.replace("_", " ").title()}')
                    axes[i].axis('off')
            
            # Show original image
            axes[7].imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
            axes[7].set_title('Original Sample')
            axes[7].axis('off')
            
            plt.tight_layout()
            plt.show()

class TexturedRenderer:
    """Enhanced renderer that supports vertex colors and proper overlay"""
    def __init__(self, focal_length, img_w, img_h, faces):
        self.focal_length = focal_length
        self.img_w = img_w
        self.img_h = img_h
        self.faces = faces
        
    def render_textured_mesh_overlay(self, vertices_list, vertex_colors_list, bg_img):
        """Render textured meshes as overlay on background image"""
        try:
            # Try to use the existing Renderer class for better integration
            from core.utils.renderer_pyrd import Renderer
            
            # Create a combined result starting with background
            result = bg_img.copy()
            
            # Render each person
            for vertices, vertex_colors in zip(vertices_list, vertex_colors_list):
                # Create a mesh with vertex colors
                mesh = trimesh.Trimesh(
                    vertices=vertices,
                    faces=self.faces,
                    vertex_colors=vertex_colors,
                    process=False
                )
                
                # Use the existing renderer but with textured mesh
                renderer = Renderer(
                    focal_length=self.focal_length, 
                    img_w=self.img_w, 
                    img_h=self.img_h, 
                    faces=self.faces,
                    same_mesh_color=False  # Use vertex colors
                )
                
                # Render with vertex colors
                textured_overlay = self.render_with_vertex_colors(
                    renderer, vertices.reshape(1, -1, 3), vertex_colors, result
                )
                result = textured_overlay
                renderer.delete()
                
            return result
            
        except Exception as e:
            print(f"Error in textured rendering: {e}")
            # Fallback to simple overlay
            return self.render_simple_overlay(vertices_list, vertex_colors_list, bg_img)
    
    def render_with_vertex_colors(self, renderer, vertices, vertex_colors, bg_img):
        """Render mesh with vertex colors using the existing renderer"""
        try:
            # Project vertices to screen coordinates
            screen_coords = self.project_vertices(vertices[0])
            
            # Create depth buffer for proper occlusion
            depth_buffer = np.full((self.img_h, self.img_w), np.inf)
            result = bg_img.copy()
            
            # Render each face
            for face_idx, face in enumerate(self.faces):
                # Get face vertices and colors
                face_vertices = vertices[0][face]
                face_colors = vertex_colors[face]
                face_screen_coords = screen_coords[face]
                
                # Check if face is visible (basic back-face culling)
                if self.is_face_visible(face_vertices):
                    # Render the face
                    self.render_face(
                        result, depth_buffer, face_screen_coords, 
                        face_vertices, face_colors
                    )
            
            return result
            
        except Exception as e:
            print(f"Error in vertex color rendering: {e}")
            # Fallback to existing renderer
            return renderer.render_front_view(vertices, bg_img_rgb=bg_img)
    
    def project_vertices(self, vertices):
        """Project 3D vertices to 2D screen coordinates"""
        # Simple perspective projection
        focal_length = float(self.focal_length)
        
        # Prevent division by zero
        z_coords = vertices[:, 2].copy()
        z_coords[z_coords == 0] = 1e-6
        
        x_2d = (vertices[:, 0] * focal_length) / z_coords + self.img_w / 2
        y_2d = (vertices[:, 1] * focal_length) / z_coords + self.img_h / 2
        
        return np.column_stack([x_2d, y_2d, z_coords])
    
    def is_face_visible(self, face_vertices):
        """Simple back-face culling"""
        # Calculate face normal
        v1 = face_vertices[1] - face_vertices[0]
        v2 = face_vertices[2] - face_vertices[0]
        normal = np.cross(v1, v2)
        
        # Face normal pointing towards camera (positive Z)
        return normal[2] > 0
    
    def render_face(self, result, depth_buffer, screen_coords, vertices_3d, colors):
        """Render a single face with vertex colors"""
        # Convert to integer pixel coordinates
        coords_2d = screen_coords[:, :2].astype(int)
        depths = vertices_3d[:, 2]
        
        # Get bounding box
        min_x = max(0, np.min(coords_2d[:, 0]))
        max_x = min(self.img_w - 1, np.max(coords_2d[:, 0]))
        min_y = max(0, np.min(coords_2d[:, 1]))
        max_y = min(self.img_h - 1, np.max(coords_2d[:, 1]))
        
        # Render using simple triangle rasterization
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                # Check if point is inside triangle
                if self.point_in_triangle([x, y], coords_2d):
                    # Calculate barycentric coordinates for interpolation
                    bary_coords = self.barycentric_coordinates([x, y], coords_2d)
                    
                    # Interpolate depth
                    depth = np.dot(bary_coords, depths)
                    
                    # Depth test
                    if depth < depth_buffer[y, x]:
                        depth_buffer[y, x] = depth
                        
                        # Interpolate color
                        color = np.dot(bary_coords, colors).astype(int)
                        color = np.clip(color, 0, 255)
                        
                        # Set pixel color (BGR format)
                        result[y, x] = [color[2], color[1], color[0]]
    
    def point_in_triangle(self, point, triangle):
        """Check if point is inside triangle using barycentric coordinates"""
        x, y = point
        x1, y1 = triangle[0]
        x2, y2 = triangle[1]
        x3, y3 = triangle[2]
        
        # Calculate barycentric coordinates
        denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if abs(denom) < 1e-10:
            return False
        
        a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
        b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
        c = 1 - a - b
        
        return a >= 0 and b >= 0 and c >= 0
    
    def barycentric_coordinates(self, point, triangle):
        """Calculate barycentric coordinates of point relative to triangle"""
        x, y = point
        x1, y1 = triangle[0]
        x2, y2 = triangle[1]
        x3, y3 = triangle[2]
        
        denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if abs(denom) < 1e-10:
            return np.array([1.0, 0.0, 0.0])
        
        a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
        b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
        c = 1 - a - b
        
        return np.array([a, b, c])
    
    def render_simple_overlay(self, vertices_list, vertex_colors_list, bg_img):
        """Simple fallback renderer using point cloud"""
        result = bg_img.copy()
        
        for vertices, vertex_colors in zip(vertices_list, vertex_colors_list):
            # Project vertices to screen coordinates
            screen_coords = self.project_vertices(vertices)
            
            # Draw points
            for i, (coord, color) in enumerate(zip(screen_coords, vertex_colors)):
                x, y = int(coord[0]), int(coord[1])
                if 0 <= x < self.img_w and 0 <= y < self.img_h:
                    # Draw small circle with vertex color
                    cv2.circle(result, (x, y), 2, (int(color[2]), int(color[1]), int(color[0])), -1)
        
        return result

class EnhancedHumanMeshEstimator(HumanMeshEstimator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.texture_transfer = TextureTransferFromSample(self.smpl_model)
    
    def process_image_with_sample_texture(self, img_path, sample_img_path, output_img_folder, i, 
                                        texture_method='spherical_mapping', smooth_colors=True,
                                        visualize_mapping=False, create_preview=True):
        """Process image and apply texture from sample image with proper overlay rendering"""
        img_cv2 = cv2.imread(str(img_path))
        sample_img = cv2.imread(str(sample_img_path))
        
        if sample_img is None:
            print(f"Could not load sample image: {sample_img_path}")
            return
        
        if visualize_mapping:
            self.texture_transfer.visualize_texture_mapping(sample_img, texture_method)
        
        fname, img_ext = os.path.splitext(os.path.basename(img_path))
        overlay_fname = os.path.join(output_img_folder, f'{os.path.basename(fname)}_{i:06d}_textured{img_ext}')
        mesh_fname = os.path.join(output_img_folder, f'{os.path.basename(fname)}_{i:06d}.obj')

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
        all_textured_vertices = []
        all_textured_colors = []
        
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
                
                # Create regular mesh for each person
                person_mesh_fname = os.path.join(output_img_folder, f'{os.path.basename(fname)}_{i:06d}_person_{person_count}.obj')
                mesh = trimesh.Trimesh(vertices_np, self.smpl_model.faces, process=False)
                mesh.export(person_mesh_fname)
                
                # Apply texture mapping
                vertex_colors = self.texture_transfer.apply_texture_to_vertices(
                    vertices_np, sample_img, method=texture_method
                )
                
                if smooth_colors:
                    vertex_colors = self.texture_transfer.smooth_vertex_colors(
                        vertex_colors, vertices_np, k=5
                    )
                
                # Transform vertices to camera space for rendering
                cam_trans = output_cam_trans[person_idx].cpu().numpy()
                vertices_cam = vertices_np + cam_trans
                
                # Store for combined rendering
                all_textured_vertices.append(vertices_cam)
                all_textured_colors.append(vertex_colors)
                
                # Create textured mesh for each person
                textured_mesh_fname = os.path.join(output_img_folder, f'{os.path.basename(fname)}_{i:06d}_person_{person_count}_textured.ply')
                textured_mesh, ply_path = self.texture_transfer.create_textured_mesh(
                    vertices_np, vertex_colors, textured_mesh_fname
                )
                
                # Create preview for each person
                if create_preview:
                    preview_path = os.path.join(output_img_folder, f'{os.path.basename(fname)}_{i:06d}_person_{person_count}_preview.png')
                    self.create_textured_preview(vertices_np, vertex_colors, preview_path)
                
                person_count += 1
                print(f"Processed person {person_count}, textured mesh saved: {ply_path}")

            # Render textured overlay with all persons using proper renderer
            focal_length = focal_length_[0]
            
            # Create textured renderer
            textured_renderer = TexturedRenderer(
                focal_length=focal_length, 
                img_w=int(img_w), 
                img_h=int(img_h), 
                faces=self.smpl_model.faces
            )
            
            # Render all textured meshes onto the original image
            textured_overlay = textured_renderer.render_textured_mesh_overlay(
                all_textured_vertices, all_textured_colors, img_cv2.copy()
            )
            
            # Save the textured overlay
            cv2.imwrite(overlay_fname, textured_overlay)
            print(f"Textured overlay saved: {overlay_fname}")
            print(f"Total persons processed: {person_count}")
    
    def create_textured_preview(self, vertices, vertex_colors, output_path):
        """Create a simple preview of the textured mesh"""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot vertices with colors
            colors_normalized = vertex_colors / 255.0
            scatter = ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                               c=colors_normalized, s=1, alpha=0.6)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Textured Mesh Preview')
            
            # Set equal aspect ratio
            max_range = np.array([vertices[:, 0].max() - vertices[:, 0].min(),
                                vertices[:, 1].max() - vertices[:, 1].min(),
                                vertices[:, 2].max() - vertices[:, 2].min()]).max() / 2.0
            mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
            mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
            mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Preview saved: {output_path}")
            
        except Exception as e:
            print(f"Could not create preview: {e}")

    def run_on_images_with_sample_texture(self, image_folder, sample_img_path, out_folder, 
                                        texture_method='spherical_mapping'):
        """Run texture transfer from sample image on all images"""
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        
        from glob import glob
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']
        images_list = [image for ext in image_extensions for image in glob(os.path.join(image_folder, ext))]
        
        print(f"Using sample texture: {sample_img_path}")
        print(f"Texture mapping method: {texture_method}")
        
        for ind, img_path in enumerate(images_list):
            print(f"Processing image {ind+1}/{len(images_list)}: {img_path}")
            self.process_image_with_sample_texture(
                img_path, sample_img_path, out_folder, ind, 
                texture_method=texture_method, 
                visualize_mapping=(ind == 0)  # Show mapping for first image only
            )

# Usage example
if __name__ == "__main__":
    import argparse
    from glob import glob
    
    def make_parser():
        parser = argparse.ArgumentParser(description='CameraHMR with Sample Texture Transfer')
        parser.add_argument("--image_folder", type=str, help="Path to input image folder.")
        parser.add_argument("--sample_texture", type=str, help="Path to sample texture image.")
        parser.add_argument("--output_folder", type=str, help="Path to output folder.")
        parser.add_argument("--texture_method", type=str, default='spherical_mapping',
                          choices=['body_parts', 'spherical_mapping', 'cylindrical_mapping'],
                          help="Texture mapping method")
        return parser

    parser = make_parser()
    args = parser.parse_args()
    
    if not args.sample_texture:
        print("Error: Please provide a sample texture image using --sample_texture")
        exit(1)
    
    # Use enhanced estimator
    estimator = EnhancedHumanMeshEstimator()
    estimator.run_on_images_with_sample_texture(
        args.image_folder, 
        args.sample_texture, 
        args.output_folder, 
        args.texture_method
    )