import numpy as np
import matplotlib.pyplot as plt

def generate_noisy_signal(timesteps=200):
    """Simulates a true signal (e.g., aircraft distance) with added Gaussian noise."""
    time = np.linspace(0, 10, timesteps)
    # True signal: Sine wave simulating changing distance
    true_signal = 10 + 5 * np.sin(time)
    # Add random Gaussian noise
    noise = np.random.normal(0, 0.8, timesteps)
    noisy_signal = true_signal + noise
    return time, true_signal, noisy_signal

def moving_average_filter(signal, window_size=5):
    """Smooths signal using a simple moving average window."""
    window = np.ones(window_size) / window_size
    smoothed_signal = np.convolve(signal, window, mode='same')
    return smoothed_signal

def low_pass_exponential_filter(signal, alpha=0.2):
    """Smooths signal using Exponential Moving Average (Low-Pass Filter)."""
    filtered = np.zeros_like(signal)
    filtered[0] = signal[0]
    for i in range(1, len(signal)):
        filtered[i] = alpha * signal[i] + (1 - alpha) * filtered[i-1]
    return filtered

def plot_and_save_results(time, true_signal, noisy_signal, ma_signal, lp_signal):
    """Plots the raw noisy data against the filtered signals and saves the image."""
    plt.figure(figsize=(10, 5))
    plt.plot(time, noisy_signal, label='Noisy Sensor Data', color='gray', alpha=0.5, linestyle='--')
    plt.plot(time, true_signal, label='True Signal', color='black', linewidth=1.5)
    plt.plot(time, ma_signal, label='Moving Average Filter (w=5)', color='blue', linewidth=2)
    plt.plot(time, lp_signal, label='Exponential Low-Pass Filter (alpha=0.2)', color='red', linewidth=2)
    
    plt.title('Sensor Signal Noise Reduction - Digital Signal Processing Demo')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Sensor Measurement')
    plt.legend()
    plt.grid(True)
    
    # Save chart image for GitHub README
    plt.savefig('signal_filtering_output.png', dpi=300, bbox_inches='tight')
    print("Plot saved successfully as 'signal_filtering_output.png'")
    plt.show()

if __name__ == "__main__":
    # Generate simulated data
    time, true_sig, noisy_sig = generate_noisy_signal()
    
    # Apply filters
    ma_filtered = moving_average_filter(noisy_sig, window_size=7)
    lp_filtered = low_pass_exponential_filter(noisy_sig, alpha=0.15)
    
    # Render and save plot
    plot_and_save_results(time, true_sig, noisy_sig, ma_filtered, lp_filtered)

