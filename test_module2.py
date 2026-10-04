# pyrefly: ignore [missing-import]
import torch
from src.model import get_segresnet_model, get_loss_function, convert_labels_to_brats_classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Instantiate Model and Loss Function
model = get_segresnet_model(in_channels=4, out_channels=3, init_filters=16).to(device)
loss_fn = get_loss_function()

# 2. Simulate a batch patch from Module 1 pipeline: (B=1, C=4, H=128, W=128, D=128)
dummy_image = torch.randn(1, 4, 128, 128, 128, dtype=torch.float32, device=device)

# Simulate raw BraTS mask with background (0) and labels (1, 2, 4)
dummy_label = torch.zeros(1, 1, 128, 128, 128, dtype=torch.long, device=device)
dummy_label[:, :, 40:80, 40:80, 40:80] = 2  # Edema
dummy_label[:, :, 50:70, 50:70, 50:70] = 1  # Core
dummy_label[:, :, 55:65, 55:65, 55:65] = 4  # Enhancing

# 3. Remap labels to 3 multi-label channels (WT, TC, ET)
target_3channel = convert_labels_to_brats_classes(dummy_label)

print("\n--- Tensor Verification ---")
print(f"Input Tensor Shape:         {dummy_image.shape}")
print(f"Multi-label Target Shape:   {target_3channel.shape}")

# 4. Forward Pass
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer.zero_grad()

outputs = model(dummy_image)
print(f"Model Output Logits Shape:  {outputs.shape}")

# 5. Compute Loss and Backward Pass
loss = loss_fn(outputs, target_3channel)
loss.backward()
optimizer.step()

print(f"Initial Dice-Focal Loss:    {loss.item():.4f}")
print("Backward pass completed successfully. Module 2 architecture verified!")