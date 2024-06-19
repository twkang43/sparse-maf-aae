import torch
import numpy as np

# Generate Sliding Windows with Stride <WIN_STRIDE>
def generate_sliding_window(x, window_size, win_stride):
    x_window = []
    for i in range(0, x.size(0)-window_size, win_stride):
        x_window.append(x[i:i+window_size]) # window_size x N

    # Extend last data if the last data is not full.
    if len(x_window[-1]) < len(x_window[-2]):
        last_data = x_window[-1]
        elements_to_add = len(x_window[-2]) - len(last_data)
        extended_last_data = last_data + last_data[:elements_to_add]
        x_window[-1] = extended_last_data
        
    return x_window
    
def reconstruct_sequence(x_hats, sequence_len, window_size, win_stride):
    x_sequence = torch.zeros(sequence_len, x_hats[0].shape[1])

    for i, x_hat in enumerate(x_hats):
        if i != 0:
            start_index = window_size + win_stride*(i-1)
            x_sequence[start_index:start_index+win_stride, :] = x_hat[-win_stride:, :]

        # Append the initial prediction array to the beginning of x_sequence.
        else:
            x_sequence[:window_size, :] = x_hat

    # Append a portion of the last prediction array to the end of x_sequence.
    x_sequence[-win_stride:, ] = x_hats[-1][-win_stride:, :]

    return x_sequence

def reconstruct_anomaly_index(anomaly_index, x_sequence, window_size, win_stride):
    anomaly_index_sequence = np.zeros_like(x_sequence)

    for i, anomaly_point in enumerate(anomaly_index):
        if i != 0:
            start_index = window_size + win_stride*(i-1)
            anomaly_index_sequence[start_index:start_index+win_stride, :] = anomaly_point[-win_stride:, :]

        # Append the initial prediction array to the beginning of anomaly_index_sequence.
        else:
            anomaly_index_sequence[:window_size, :] = anomaly_point

    # Append a portion of the last prediction array to the end of x_sequence.
    anomaly_index_sequence[-win_stride:, ] = anomaly_index[-1][-win_stride:, :]

    return anomaly_index_sequence