"""
Interactive point selection module with enhanced visualization.
"""

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageTk
import numpy as np
from .config import Config


class PointSelector:
    """Interactive point selector with object and depth visualization."""
    
    def __init__(self, environment_image, detected_objects=None, depth_data=None):
        self.config = Config()
        self.environment_image = environment_image
        self.detected_objects = detected_objects or []
        self.depth_data = depth_data
        
        # GUI components
        self.root = None
        self.canvas = None
        self.photo = None
        
        # Selection state
        self.selected_point = None
        self.show_objects = True
        self.show_depth = False
        
        # Prepare display image
        self.display_image, self.scale_factor = self._prepare_display_image()
        self.overlay_image = self._create_overlay_image()
    
    def _prepare_display_image(self):
        """Prepare image for display with proper scaling."""
        original_width, original_height = self.environment_image.size
        
        # Calculate scaling factor
        max_width = self.config.max_display_width
        max_height = self.config.max_display_height
        
        width_ratio = max_width / original_width
        height_ratio = max_height / original_height
        scale_factor = min(width_ratio, height_ratio, 1.0)  # Don't upscale
        
        if scale_factor < 1.0:
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            display_image = self.environment_image.resize(
                (new_width, new_height), 
                Image.Resampling.LANCZOS
            )
        else:
            display_image = self.environment_image.copy()
            scale_factor = 1.0
        
        return display_image, scale_factor
    
    def _create_overlay_image(self):
        """Create overlay with detected objects and depth visualization."""
        overlay = self.display_image.copy()
        draw = ImageDraw.Draw(overlay)
        
        # Draw detected objects if available
        if self.show_objects and self.detected_objects:
            self._draw_detected_objects(draw)
        
        # Add depth visualization if available
        if self.show_depth and self.depth_data is not None:
            overlay = self._blend_with_depth(overlay)
        
        return overlay
    
    def _draw_detected_objects(self, draw):
        """Draw detected objects on the overlay."""
        colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
        
        for i, obj in enumerate(self.detected_objects):
            bbox = obj['bbox']
            class_name = obj['class_name']
            confidence = obj['confidence']
            
            # Scale bounding box to display coordinates
            scaled_bbox = [
                int(bbox[0] * self.scale_factor),
                int(bbox[1] * self.scale_factor),
                int(bbox[2] * self.scale_factor),
                int(bbox[3] * self.scale_factor)
            ]
            
            color = colors[i % len(colors)]
            
            # Draw bounding box
            draw.rectangle(scaled_bbox, outline=color, width=2)
            
            # Draw label with background
            label = f"{class_name} ({confidence:.2f})"
            
            # Get text size (approximate)
            text_width = len(label) * 7  # Approximate width
            text_height = 15
            
            # Draw label background
            label_bg = [
                scaled_bbox[0], 
                scaled_bbox[1] - text_height - 4,
                scaled_bbox[0] + text_width + 8, 
                scaled_bbox[1]
            ]
            draw.rectangle(label_bg, fill=color)
            
            # Draw text
            draw.text(
                (scaled_bbox[0] + 4, scaled_bbox[1] - text_height - 2), 
                label, 
                fill='white'
            )
            
            # Add depth info if available
            if 'depth_info' in obj:
                depth_info = obj['depth_info']
                depth_category = depth_info.get('depth_category', 'unknown')
                depth_text = f"Depth: {depth_category}"
                draw.text(
                    (scaled_bbox[0] + 4, scaled_bbox[1] + 4),
                    depth_text,
                    fill=color
                )
    
    def _blend_with_depth(self, overlay):
        """Blend overlay with depth visualization."""
        if self.depth_data is None or 'normalized_depth' not in self.depth_data:
            return overlay
        
        try:
            # Get depth map and resize to match display
            depth_map = self.depth_data['normalized_depth']
            depth_pil = Image.fromarray(depth_map, mode='L')
            
            # Resize depth map to match display image
            depth_resized = depth_pil.resize(
                self.display_image.size, 
                Image.Resampling.LANCZOS
            )
            
            # Convert to RGB and apply colormap effect
            depth_rgb = Image.new('RGB', depth_resized.size)
            depth_array = np.array(depth_resized)
            
            # Create a blue-to-red depth visualization
            depth_colored = np.zeros((depth_array.shape[0], depth_array.shape[1], 3), dtype=np.uint8)
            depth_colored[:, :, 0] = depth_array  # Red channel for far objects
            depth_colored[:, :, 2] = 255 - depth_array  # Blue channel for near objects
            
            depth_rgb = Image.fromarray(depth_colored)
            
            # Blend with original overlay (30% depth, 70% original)
            blended = Image.blend(overlay, depth_rgb, alpha=0.3)
            return blended
            
        except Exception as e:
            print(f"⚠️ Depth visualization error: {e}")
            return overlay
    
    def _convert_display_to_original_coords(self, display_x, display_y):
        """Convert display coordinates to original image coordinates."""
        original_x = int(display_x / self.scale_factor)
        original_y = int(display_y / self.scale_factor)
        
        # Ensure within bounds
        original_x = max(0, min(original_x, self.environment_image.size[0] - 1))
        original_y = max(0, min(original_y, self.environment_image.size[1] - 1))
        
        return original_x, original_y
    
    def _on_click(self, event):
        """Handle mouse click on image."""
        # Convert to original coordinates
        original_x, original_y = self._convert_display_to_original_coords(event.x, event.y)
        
        # Store selection
        self.selected_point = (original_x, original_y)
        
        # Update visual feedback
        self._update_selection_display(event.x, event.y, original_x, original_y)
        
        print(f"✅ Point selected: ({original_x}, {original_y})")
        
        # Show depth info if available
        if self.depth_data is not None:
            depth_info = self._get_depth_info_at_point(original_x, original_y)
            if depth_info:
                print(f"   Depth category: {depth_info['depth_category']}")
                print(f"   Relative depth: {depth_info['relative_depth']:.2f}")
    
    def _get_depth_info_at_point(self, x, y):
        """Get depth information at the selected point."""
        if self.depth_data is None or 'raw_depth' not in self.depth_data:
            return None
        
        try:
            from .depth_estimator import DepthEstimator
            estimator = DepthEstimator()
            return estimator.get_depth_at_point(self.depth_data, x, y)
        except Exception as e:
            print(f"⚠️ Error getting depth info: {e}")
            return None
    
    def _update_selection_display(self, display_x, display_y, original_x, original_y):
        """Update the visual display with selection marker."""
        # Clear previous markers
        self.canvas.delete("marker")
        
        # Draw selection marker
        radius = 8
        self.canvas.create_oval(
            display_x - radius, display_y - radius,
            display_x + radius, display_y + radius,
            fill='red', outline='darkred', width=3, tags="marker"
        )
        
        # Draw crosshair
        self.canvas.create_line(
            display_x - radius*2, display_y,
            display_x + radius*2, display_y,
            fill='red', width=2, tags="marker"
        )
        self.canvas.create_line(
            display_x, display_y - radius*2,
            display_x, display_y + radius*2,
            fill='red', width=2, tags="marker"
        )
        
        # Update status text
        status_text = f"Selected: ({original_x}, {original_y})"
        
        # Add depth info if available
        if self.depth_data is not None:
            depth_info = self._get_depth_info_at_point(original_x, original_y)
            if depth_info:
                status_text += f" | Depth: {depth_info['depth_category']}"
        
        self.canvas.create_text(
            10, 10, 
            text=status_text,
            fill='red', 
            font=('Arial', 12, 'bold'), 
            anchor='nw', 
            tags="marker"
        )
    
    def _toggle_objects(self):
        """Toggle object detection visualization."""
        self.show_objects = not self.show_objects
        self._refresh_display()
    
    def _toggle_depth(self):
        """Toggle depth visualization."""
        self.show_depth = not self.show_depth
        self._refresh_display()
    
    def _refresh_display(self):
        """Refresh the display with updated overlay."""
        self.overlay_image = self._create_overlay_image()
        self.photo = ImageTk.PhotoImage(self.overlay_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # Redraw selection marker if exists
        if self.selected_point:
            display_x = int(self.selected_point[0] * self.scale_factor)
            display_y = int(self.selected_point[1] * self.scale_factor)
            self._update_selection_display(
                display_x, display_y, 
                self.selected_point[0], self.selected_point[1]
            )
    
    def select_point(self):
        """Open the point selection interface."""
        self.root = tk.Tk()
        self.root.title("Select Object Placement Point")
        self.root.resizable(False, False)
        
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10)
        
        # Instructions
        instruction_text = (
            "Click on the environment image to select where you want to place the object.\n"
            "Red boxes show detected objects. Use toggle buttons to show/hide overlays."
        )
        instruction_label = tk.Label(
            main_frame,
            text=instruction_text,
            font=('Arial', 11),
            wraplength=800,
            justify='left'
        )
        instruction_label.pack(pady=(0, 10))
        
        # Control buttons frame
        controls_frame = tk.Frame(main_frame)
        controls_frame.pack(pady=(0, 10))
        
        # Toggle buttons
        if self.detected_objects:
            objects_btn = tk.Button(
                controls_frame,
                text="Toggle Objects",
                command=self._toggle_objects,
                bg='lightblue',
                font=('Arial', 10)
            )
            objects_btn.pack(side=tk.LEFT, padx=5)
        
        if self.depth_data is not None:
            depth_btn = tk.Button(
                controls_frame,
                text="Toggle Depth",
                command=self._toggle_depth,
                bg='lightgreen',
                font=('Arial', 10)
            )
            depth_btn.pack(side=tk.LEFT, padx=5)
        
        # Canvas for image display
        self.canvas = tk.Canvas(
            main_frame,
            width=self.overlay_image.size[0],
            height=self.overlay_image.size[1],
            bg='white'
        )
        self.canvas.pack()
        
        # Display image
        self.photo = ImageTk.PhotoImage(self.overlay_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # Bind click event
        self.canvas.bind("<Button-1>", self._on_click)
        
        # Action buttons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        confirm_btn = tk.Button(
            button_frame,
            text="Confirm Selection",
            command=self._confirm_selection,
            bg='lightgreen',
            font=('Arial', 12),
            padx=20
        )
        confirm_btn.pack(side=tk.LEFT, padx=10)
        
        cancel_btn = tk.Button(
            button_frame,
            text="Cancel",
            command=self._cancel_selection,
            bg='lightcoral',
            font=('Arial', 12),
            padx=20
        )
        cancel_btn.pack(side=tk.LEFT, padx=10)
        
        # Info panel
        info_text = f"Image: {self.environment_image.size[0]}×{self.environment_image.size[1]} pixels"
        if self.scale_factor < 1.0:
            info_text += f" (scaled to {self.overlay_image.size[0]}×{self.overlay_image.size[1]} for display)"
        
        if self.detected_objects:
            info_text += f" | Objects: {len(self.detected_objects)} detected"
        
        if self.depth_data is not None:
            info_text += " | Depth: Available"
        
        info_label = tk.Label(
            main_frame, 
            text=info_text, 
            font=('Arial', 9), 
            fg='gray'
        )
        info_label.pack(pady=(10, 0))
        
        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f"+{x}+{y}")
        
        # Start GUI
        self.root.mainloop()
        
        return self.selected_point
    
    def _confirm_selection(self):
        """Confirm the selected point."""
        if self.selected_point:
            self.root.quit()
            self.root.destroy()
        else:
            messagebox.showwarning(
                "No Selection", 
                "Please click on the image to select a point first."
            )
    
    def _cancel_selection(self):
        """Cancel point selection."""
        self.selected_point = None
        self.root.quit()
        self.root.destroy()