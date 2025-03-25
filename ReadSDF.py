import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def readfile(path):
    data = {}
    current_frequency = None
    tau_array = None
    num_lines = None

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith('ACQ = '):
            acquisitionTime = line.split(' = ', 1)[1]
            num_lines = int(float(acquisitionTime)) * 10

        elif line.startswith('TAU = ['):
            tau_step_line = line.split(' = ', 1)[1].strip('[]')
            max_tau = float(tau_step_line.split(':')[1].split('*')[0])
            min_tau = float(tau_step_line.split(':')[2].split('*')[0])
            number_of_taus = int(tau_step_line.split(':')[3])

        elif line.startswith('BR = {'):
            BR = line.split(' = ', 1)[1].strip('{}')
            valueFrequencies = np.fromstring(BR, sep=',')
            amountFrequencies = len(valueFrequencies)

        elif line.startswith('BR = '):
            current_frequency = line.split(' = ')[1]
            if current_frequency not in data:
                data[current_frequency] = {}
            else:
                current_frequency += f'_{i}'
                data[current_frequency] = {}

        elif line.startswith('T1MAX = '):
            T1_estimated = float(line.split(' = ')[1])
            try:
                tau_array = np.logspace(np.log10(min_tau * T1_estimated), np.log10(max_tau * T1_estimated), num=number_of_taus)[::-1]
                for tau in tau_array:
                    data[current_frequency][tau] = None  # Placeholder
            except:
                print("The TAU line wasn't read yet; skipping.")

        elif line.startswith('REAL') and tau_array is not None:
            i += 1  # Move to the first data line after REAL
            for tau in tau_array:
                time = []
                magnitude = []

                for _ in range(num_lines):
                    if i >= len(lines):
                        break
                    columns = lines[i].strip().split()
                    if len(columns) >= 4:
                        magnitude.append(float(columns[2]))
                        time.append(float(columns[3]))
                    i += 1

                df = pd.DataFrame({'Time': time, 'Magnitude': magnitude})
                df['Mean'] = df['Magnitude'].mean()
                data[current_frequency][tau] = df

        i += 1
    return data

def plot_fid_and_kill_your_computer(data):
    for frequency, taus in data.items():
        plt.figure(figsize=(10, 6))
        plt.title(f"Frequency: {frequency} Hz")
        plt.xlabel('Time (μs)')
        plt.ylabel('Magnitude')

        for tau, df in taus.items():
            if df is not None:
                plt.plot(df['Time'], df['Magnitude'], label=f"T1 = {tau:.2f}", alpha=0.7)

        plt.legend()
        plt.tight_layout()
        plt.show()

def extract_t1_curves(data, save_dir):
    for frequency, taus in data.items():
        tau_values = []
        mean_values = []

        for tau, df in taus.items():
            if df is not None:
                tau_values.append(tau)
                mean_values.append(df['Mean'].iloc[0])

        df_t1 = pd.DataFrame({'Tau': tau_values, 'Mean Magnitude': mean_values})
        df_t1.sort_values(by='Tau', inplace=True)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.title(f"Mean Magnitude vs Tau for Frequency: {frequency} Hz")
        plt.xlabel('Tau (μs)')
        plt.ylabel('Mean Magnitude')
        plt.plot(df_t1['Tau'], df_t1['Mean Magnitude'], marker='o', alpha=0.7)
        plt.tight_layout()

        # Save plot
        plot_filename = os.path.join(save_dir, f"t1_curve_{frequency.replace('.', '_')}.png")
        plt.savefig(plot_filename)
        plt.close()

        # Save data
        data_filename = os.path.join(save_dir, f"T1_{frequency}.dat")
        df_t1.to_csv(data_filename, sep='\t', index=False, float_format='%.6f')

def exponential_decay(x, a, b, c):
    return a * np.exp(-x / b) + c

def fit_t1_to_data(data):
    for frequency, taus in data.items():
        tau_values = []
        mean_values = []

        for tau, df in taus.items():
            if df is not None:
                tau_values.append(tau)
                mean_values.append(df['Mean'].iloc[0])

        try:
            x = np.array(tau_values)
            y = np.array(mean_values)
            initial_guess = [max(y), np.mean(x)/3, min(y)]
            popt, _ = curve_fit(exponential_decay, x, y, p0=initial_guess)
            T1 = popt[1]
            data[frequency]['T1'] = T1
        except Exception as e:
            print(f"Could not fit for frequency {frequency} Hz: {e}")
    return data

def plot_frequency_vs_T1(data):
    freq_list = []
    t1_list = []

    for frequency, taus in data.items():
        if 'T1' in taus:
            freq_list.append(float(frequency))
            t1_list.append(taus['T1'])

    df = pd.DataFrame({'Frequency': freq_list, 'T1': t1_list})
    df.sort_values(by='Frequency', inplace=True)

    plt.figure(figsize=(10, 6))
    plt.plot(df['Frequency'], df['T1'], marker='o', linestyle='-', color='b')
    plt.title('Frequency vs T1')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('T1 (μs)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Process NMRD data with pandas.")
    parser.add_argument('path', type=str, help="Path to the .sdf file")

    args = parser.parse_args()
    file_path = args.path
    save_dir = os.path.dirname(file_path)

    data = readfile(file_path)

    # Optional heavy plotting
    # plot_fid_and_kill_your_computer(data)

    extract_t1_curves(data, save_dir)
    fit_t1_to_data(data)
    plot_frequency_vs_T1(data)

if __name__ == '__main__':
    main()
