import torch
import torch.nn as nn

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("trained_models/MTASR_pa0.75_hr1_10_2100.pth", map_location=device, weights_only=False)
model.to(device)
model.eval()  # Set to evaluation mode

# Function to run inference
def run_inference(bvp_signal, net_type="both"):
    """
    Runs inference on a single BVP signal.

    :param bvp_signal: NumPy array or PyTorch tensor of shape (1, 2100)
    :param net_type: "hr" for heart rate estimation, "both" for classification & HR
    :return: Model output (classification, HR, peaks depending on net_type)
    """
    # Convert input to tensor if necessary
    if isinstance(bvp_signal, np.ndarray):
        bvp_signal = torch.tensor(bvp_signal, dtype=torch.float32)

    # Ensure correct shape (batch_size=1, channels=1, sequence_length=2100)
    bvp_signal = bvp_signal.unsqueeze(0).unsqueeze(0).to(device)  # Shape: (1, 1, 2100)

    # Run inference
    with torch.no_grad():
        if net_type == "hr":
            hr, p_peak = model(bvp_signal, net_type)
            return hr.cpu().numpy(), p_peak.cpu().numpy()
        elif net_type == "both":
            outputs, hr, p_peak = model(bvp_signal, net_type)
            predicted_class = torch.argmax(outputs, dim=1).item()
            return predicted_class, hr.cpu().numpy(), p_peak.cpu().numpy()

# Example usage
import numpy as np

# svaki subject ima 3 videa, svaki video zauzima 4 reda u data
data = np.load(r"ubfc_phys_new.npy", allow_pickle=True)
bvp_signal = data[10][4] #obican niz od 2100 elemenata #4. kolona je rppg
# print(bvp_signal.shape)
# print(bvp_signal)

result = run_inference(bvp_signal, net_type="both")
print(result)

# for sample in range(data.shape[0]):
#     bvp_signal = data[sample][4]
#     result = run_inference(bvp_signal, net_type="both")
#     print(sample, data[sample][0], result[0])
