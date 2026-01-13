import torch
import torchvision

# Télécharger et sauvegarder les poids
print('Start')
model = torchvision.models.vgg16(pretrained=True)
torch.save(model.state_dict(), "./utils_test/vgg16_weights.pth")
print('Done')