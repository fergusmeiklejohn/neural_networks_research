"""
Physics simulation environment for Experiment 1: Simple Physics Worlds
Uses pymunk for 2D physics simulation with configurable rules.
"""

import numpy as np
import pymunk
import pygame
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import time


@dataclass
class PhysicsConfig:
    """Configuration for physics simulation parameters"""
    gravity: float = -981.0  # Gravity in pixels/s^2 (negative = downward)
    friction: float = 0.7    # Surface friction coefficient
    elasticity: float = 0.8  # Bounce coefficient
    damping: float = 0.95    # Air resistance/damping
    world_width: int = 800   # World dimensions in pixels
    world_height: int = 600
    dt: float = 1/60         # Time step (60 FPS)
    

@dataclass
class Ball:
    """Ball object with physical properties"""
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 20.0
    mass: float = 1.0
    color: Tuple[int, int, int] = (255, 0, 0)


class PhysicsWorld:
    """2D Physics world simulation using Pymunk"""
    
    def __init__(self, config: PhysicsConfig):
        self.config = config
        self.space = pymunk.Space()
        self.space.gravity = (0, config.gravity)
        self.space.damping = config.damping
        
        # Create boundaries
        self._create_boundaries()
        
        # Storage for trajectories
        self.trajectory_data = []
        self.balls = []
        
    def _create_boundaries(self):
        """Create world boundaries (walls)"""
        # Floor
        floor = pymunk.Segment(self.space.static_body, 
                              (0, 0), 
                              (self.config.world_width, 0), 5)
        floor.friction = self.config.friction
        floor.elasticity = self.config.elasticity
        
        # Ceiling  
        ceiling = pymunk.Segment(self.space.static_body,
                                (0, self.config.world_height),
                                (self.config.world_width, self.config.world_height), 5)
        ceiling.friction = self.config.friction
        ceiling.elasticity = self.config.elasticity
        
        # Left wall
        left_wall = pymunk.Segment(self.space.static_body,
                                  (0, 0),
                                  (0, self.config.world_height), 5)
        left_wall.friction = self.config.friction
        left_wall.elasticity = self.config.elasticity
        
        # Right wall
        right_wall = pymunk.Segment(self.space.static_body,
                                   (self.config.world_width, 0),
                                   (self.config.world_width, self.config.world_height), 5)
        right_wall.friction = self.config.friction
        right_wall.elasticity = self.config.elasticity
        
        self.space.add(floor, ceiling, left_wall, right_wall)
    
    def add_ball(self, ball: Ball) -> pymunk.Body:
        """Add a ball to the simulation"""
        # Create body and shape
        body = pymunk.Body(ball.mass, pymunk.moment_for_circle(ball.mass, 0, ball.radius))
        body.position = ball.x, ball.y
        body.velocity = ball.vx, ball.vy
        
        shape = pymunk.Circle(body, ball.radius)
        shape.friction = self.config.friction
        shape.elasticity = self.config.elasticity
        
        self.space.add(body, shape)
        self.balls.append((body, shape, ball))
        
        return body
    
    def modify_physics(self, modifications: Dict[str, float]):
        """Modify physics parameters during simulation"""
        if 'gravity' in modifications:
            self.space.gravity = (0, modifications['gravity'])
            
        if 'friction' in modifications:
            # Update friction for all objects
            for shape in self.space.shapes:
                if hasattr(shape, 'friction'):
                    shape.friction = modifications['friction']
                    
        if 'elasticity' in modifications:
            # Update elasticity for all objects
            for shape in self.space.shapes:
                if hasattr(shape, 'elasticity'):
                    shape.elasticity = modifications['elasticity']
                    
        if 'damping' in modifications:
            self.space.damping = modifications['damping']
    
    def step(self, steps: int = 1) -> List[Dict[str, Any]]:
        """Step the simulation forward and record trajectory data"""
        frame_data = []
        
        for _ in range(steps):
            # Record current state
            current_frame = {
                'time': len(self.trajectory_data) * self.config.dt,
                'balls': []
            }
            
            for body, shape, ball_info in self.balls:
                ball_state = {
                    'id': id(body),
                    'x': body.position.x,
                    'y': body.position.y,
                    'vx': body.velocity.x,
                    'vy': body.velocity.y,
                    'radius': ball_info.radius,
                    'mass': ball_info.mass,
                    'kinetic_energy': 0.5 * ball_info.mass * (body.velocity.x**2 + body.velocity.y**2),
                    'potential_energy': ball_info.mass * abs(self.space.gravity[1]) * body.position.y
                }
                current_frame['balls'].append(ball_state)
            
            frame_data.append(current_frame)
            
            # Step physics
            self.space.step(self.config.dt)
        
        self.trajectory_data.extend(frame_data)
        return frame_data
    
    def get_trajectory_array(self) -> np.ndarray:
        """Convert trajectory data to numpy array for neural network training"""
        if not self.trajectory_data:
            return np.array([])
        
        # Extract features: [time, x, y, vx, vy, mass, radius, ke, pe] for each ball
        features = []
        
        for frame in self.trajectory_data:
            frame_features = [frame['time']]
            for ball in frame['balls']:
                frame_features.extend([
                    ball['x'], ball['y'], ball['vx'], ball['vy'],
                    ball['mass'], ball['radius'], 
                    ball['kinetic_energy'], ball['potential_energy']
                ])
            features.append(frame_features)
        
        return np.array(features)
    
    def visualize_trajectory(self, save_path: Optional[str] = None):
        """Visualize ball trajectories"""
        if not self.trajectory_data:
            print("No trajectory data to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract data for plotting
        times = [frame['time'] for frame in self.trajectory_data]
        
        for ball_idx, (_, _, ball_info) in enumerate(self.balls):
            ball_id = id(self.balls[ball_idx][0])
            
            positions_x = []
            positions_y = []
            velocities_x = []
            velocities_y = []
            energies_k = []
            energies_p = []
            
            for frame in self.trajectory_data:
                for ball_data in frame['balls']:
                    if ball_data['id'] == ball_id:
                        positions_x.append(ball_data['x'])
                        positions_y.append(ball_data['y'])
                        velocities_x.append(ball_data['vx'])
                        velocities_y.append(ball_data['vy'])
                        energies_k.append(ball_data['kinetic_energy'])
                        energies_p.append(ball_data['potential_energy'])
                        break
            
            # Plot trajectory
            axes[0, 0].plot(positions_x, positions_y, 
                           label=f'Ball {ball_idx+1}', linewidth=2)
            
            # Plot velocity over time
            axes[0, 1].plot(times[:len(velocities_x)], velocities_x, 
                           label=f'Ball {ball_idx+1} vx', alpha=0.7)
            axes[0, 1].plot(times[:len(velocities_y)], velocities_y, 
                           label=f'Ball {ball_idx+1} vy', alpha=0.7, linestyle='--')
            
            # Plot energy over time
            axes[1, 0].plot(times[:len(energies_k)], energies_k, 
                           label=f'Ball {ball_idx+1} KE', alpha=0.7)
            axes[1, 0].plot(times[:len(energies_p)], energies_p, 
                           label=f'Ball {ball_idx+1} PE', alpha=0.7, linestyle='--')
            
            # Total energy
            total_energy = [k + p for k, p in zip(energies_k, energies_p)]
            axes[1, 1].plot(times[:len(total_energy)], total_energy, 
                           label=f'Ball {ball_idx+1} Total', alpha=0.7)
        
        # Configure plots
        axes[0, 0].set_title('Trajectory (Position)')
        axes[0, 0].set_xlabel('X Position')
        axes[0, 0].set_ylabel('Y Position')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Velocity vs Time')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Velocity')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Energy Components vs Time')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Energy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Total Energy vs Time')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Total Energy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def reset(self):
        """Reset the simulation"""
        # Remove all balls
        for body, shape, _ in self.balls:
            self.space.remove(body, shape)
        
        self.balls.clear()
        self.trajectory_data.clear()
    
    def get_physics_metrics(self) -> Dict[str, float]:
        """Calculate physics validation metrics"""
        if not self.trajectory_data:
            return {}
        
        metrics = {}
        
        # Energy conservation check
        total_energies = []
        for frame in self.trajectory_data:
            total_energy = sum(ball['kinetic_energy'] + ball['potential_energy'] 
                             for ball in frame['balls'])
            total_energies.append(total_energy)
        
        if len(total_energies) > 1:
            energy_variance = np.var(total_energies)
            metrics['energy_conservation'] = 1.0 / (1.0 + energy_variance)
        
        # Trajectory smoothness
        for ball_idx, (_, _, _) in enumerate(self.balls):
            ball_id = id(self.balls[ball_idx][0])
            velocities = []
            
            for frame in self.trajectory_data:
                for ball_data in frame['balls']:
                    if ball_data['id'] == ball_id:
                        velocities.append([ball_data['vx'], ball_data['vy']])
                        break
            
            if len(velocities) > 2:
                velocities = np.array(velocities)
                accelerations = np.diff(velocities, axis=0) / self.config.dt
                smoothness = 1.0 / (1.0 + np.mean(np.sum(accelerations**2, axis=1)))
                metrics[f'ball_{ball_idx}_smoothness'] = smoothness
        
        return metrics


def create_sample_scenario() -> Tuple[PhysicsWorld, List[Ball]]:
    """Create a sample physics scenario for testing"""
    config = PhysicsConfig()
    world = PhysicsWorld(config)
    
    # Create some balls
    balls = [
        Ball(x=100, y=500, vx=150, vy=0, radius=15),
        Ball(x=200, y=400, vx=-100, vy=100, radius=20),
        Ball(x=300, y=450, vx=0, vy=-50, radius=18)
    ]
    
    for ball in balls:
        world.add_ball(ball)
    
    return world, balls


if __name__ == "__main__":
    # Test the physics environment
    print("Testing Physics Environment...")
    
    world, balls = create_sample_scenario()
    
    print(f"Created world with {len(balls)} balls")
    print("Running simulation for 5 seconds...")
    
    # Run simulation
    for _ in range(300):  # 5 seconds at 60 FPS
        world.step()
    
    print(f"Simulation complete. Generated {len(world.trajectory_data)} frames of data")
    
    # Get metrics
    metrics = world.get_physics_metrics()
    print("Physics Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Visualize
    world.visualize_trajectory()
    
    # Test physics modification
    print("\nTesting physics modification...")
    world.reset()
    
    # Add balls again
    for ball in balls:
        world.add_ball(ball)
    
    # Run with modified gravity
    world.modify_physics({'gravity': -1500})  # 50% stronger gravity
    
    for _ in range(300):
        world.step()
    
    print("Modified physics simulation complete")
    world.visualize_trajectory()