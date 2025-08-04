"""
Test pendulum data generation with small dataset
"""

import matplotlib.pyplot as plt
import numpy as np
from pendulum_data_generator import (
    PendulumConfig,
    PendulumDataConfig,
    PendulumDataGenerator,
    PendulumSimulator,
)


def test_pendulum_physics():
    """Test basic pendulum physics simulation"""
    print("Testing pendulum physics simulation...")

    # Test fixed-length pendulum
    config_fixed = PendulumConfig(length=1.0, gravity=9.8, damping=0.01)
    simulator_fixed = PendulumSimulator(config_fixed)

    # Small angle initial condition
    theta0 = np.pi / 12  # 15 degrees
    theta_dot0 = 0.0

    trajectory = simulator_fixed.simulate(theta0, theta_dot0, duration=5.0)

    # Check energy conservation (should be approximately constant for fixed length)
    energy = trajectory["energy"]
    energy_var = np.std(energy) / np.mean(np.abs(energy))
    print(f"Fixed pendulum energy variation: {energy_var:.3f} (should be < 0.15)")

    # Test time-varying pendulum
    config_varying = PendulumConfig(
        length=1.0,
        gravity=9.8,
        damping=0.01,
        length_variation=0.2,
        length_frequency=0.1,
    )
    simulator_varying = PendulumSimulator(config_varying)

    trajectory_varying = simulator_varying.simulate(theta0, theta_dot0, duration=5.0)

    # Energy should NOT be conserved for varying length
    energy_varying = trajectory_varying["energy"]
    energy_var_varying = np.std(energy_varying) / np.mean(np.abs(energy_varying))
    print(
        f"Varying pendulum energy variation: {energy_var_varying:.3f} (should be > 0.02)"
    )

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Fixed pendulum trajectory
    axes[0, 0].plot(trajectory["x"], trajectory["y"], "b-", alpha=0.5)
    axes[0, 0].scatter(trajectory["x"][::10], trajectory["y"][::10], c="blue", s=20)
    axes[0, 0].set_title("Fixed Length Pendulum")
    axes[0, 0].set_xlabel("x (m)")
    axes[0, 0].set_ylabel("y (m)")
    axes[0, 0].axis("equal")

    # Varying pendulum trajectory
    axes[0, 1].plot(trajectory_varying["x"], trajectory_varying["y"], "r-", alpha=0.5)
    axes[0, 1].scatter(
        trajectory_varying["x"][::10], trajectory_varying["y"][::10], c="red", s=20
    )
    axes[0, 1].set_title("Time-Varying Length Pendulum")
    axes[0, 1].set_xlabel("x (m)")
    axes[0, 1].set_ylabel("y (m)")
    axes[0, 1].axis("equal")

    # Angular position over time
    axes[1, 0].plot(trajectory["t"], trajectory["theta"], "b-", label="Fixed")
    axes[1, 0].plot(
        trajectory_varying["t"], trajectory_varying["theta"], "r-", label="Varying"
    )
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Angle (rad)")
    axes[1, 0].legend()
    axes[1, 0].set_title("Angular Position")

    # Length and energy
    axes[1, 1].plot(
        trajectory_varying["t"], trajectory_varying["length"], "g-", label="Length"
    )
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Length (m)", color="g")
    axes[1, 1].tick_params(axis="y", labelcolor="g")

    ax2 = axes[1, 1].twinx()
    ax2.plot(
        trajectory_varying["t"], trajectory_varying["energy"], "orange", label="Energy"
    )
    ax2.set_ylabel("Energy", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")
    axes[1, 1].set_title("Time-Varying Length and Energy")

    plt.tight_layout()
    plt.savefig("pendulum_physics_test.png", dpi=150)
    print("Saved physics test visualization to pendulum_physics_test.png")

    return energy_var < 0.15 and energy_var_varying > 0.02


def test_data_generation():
    """Test data generation with small dataset"""
    print("\nTesting data generation...")

    # Small test configuration
    config = PendulumDataConfig(
        num_samples=100,  # Small for testing
        sequence_length=120,  # 2 seconds
        output_dir="data/processed/pendulum_test",
    )

    generator = PendulumDataGenerator(config)

    # Generate small datasets
    print("Generating test fixed-length data...")
    fixed_data = generator.generate_dataset(mechanism="fixed", num_samples=50)

    print("Generating test time-varying data...")
    varying_data = generator.generate_dataset(mechanism="time-varying", num_samples=50)

    # Analyze datasets
    print(f"\nFixed dataset: {len(fixed_data['trajectories'])} trajectories")
    print(f"Varying dataset: {len(varying_data['trajectories'])} trajectories")

    # Compare statistics
    fixed_energies = []
    varying_energies = []

    for traj in fixed_data["trajectories"]:
        energy = traj["trajectory"]["energy"]
        energy_var = np.std(energy) / np.mean(np.abs(energy))
        fixed_energies.append(energy_var)

    for traj in varying_data["trajectories"]:
        energy = traj["trajectory"]["energy"]
        energy_var = np.std(energy) / np.mean(np.abs(energy))
        varying_energies.append(energy_var)

    print(
        f"\nFixed pendulum energy variation: {np.mean(fixed_energies):.3f} ± {np.std(fixed_energies):.3f}"
    )
    print(
        f"Varying pendulum energy variation: {np.mean(varying_energies):.3f} ± {np.std(varying_energies):.3f}"
    )

    # Save test datasets
    generator.save_dataset(fixed_data, "test_fixed")
    generator.save_dataset(varying_data, "test_varying")

    return True


def visualize_mechanism_shift():
    """Visualize the mechanism shift effect"""
    print("\nVisualizing mechanism shift...")

    # Same initial conditions
    theta0 = np.pi / 8
    theta_dot0 = 0.5

    # Fixed vs varying configs
    config_fixed = PendulumConfig(length=1.0, gravity=9.8, damping=0.01)
    config_varying = PendulumConfig(
        length=1.0,
        gravity=9.8,
        damping=0.01,
        length_variation=0.2,
        length_frequency=0.1,
    )

    sim_fixed = PendulumSimulator(config_fixed)
    sim_varying = PendulumSimulator(config_varying)

    traj_fixed = sim_fixed.simulate(theta0, theta_dot0, 10.0)
    traj_varying = sim_varying.simulate(theta0, theta_dot0, 10.0)

    # Create phase space plot
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(
        traj_fixed["theta"],
        traj_fixed["theta_dot"],
        "b-",
        alpha=0.7,
        label="Fixed Length",
    )
    plt.plot(
        traj_varying["theta"],
        traj_varying["theta_dot"],
        "r-",
        alpha=0.7,
        label="Varying Length",
    )
    plt.xlabel("θ (rad)")
    plt.ylabel("θ̇ (rad/s)")
    plt.title("Phase Space Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    # Show divergence over time
    position_diff = np.abs(traj_fixed["theta"] - traj_varying["theta"])
    plt.semilogy(traj_fixed["t"], position_diff)
    plt.xlabel("Time (s)")
    plt.ylabel("|θ_fixed - θ_varying| (rad)")
    plt.title("Trajectory Divergence (log scale)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("pendulum_mechanism_shift.png", dpi=150)
    print("Saved mechanism shift visualization to pendulum_mechanism_shift.png")


def main():
    """Run all tests"""
    print("Running pendulum data generation tests...\n")

    # Test physics
    physics_ok = test_pendulum_physics()
    print(f"Physics test: {'PASSED' if physics_ok else 'FAILED'}")

    # Test data generation
    generation_ok = test_data_generation()
    print(f"Generation test: {'PASSED' if generation_ok else 'FAILED'}")

    # Visualize mechanism shift
    visualize_mechanism_shift()

    print("\nAll tests complete!")


if __name__ == "__main__":
    main()
