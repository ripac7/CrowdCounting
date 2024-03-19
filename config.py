
# Parameters
norm = 100
learning_rate= 0.000001
batch_size = 24
num_epochs = 50
momentum = 0.9

# Paths
folder_path = './processed_data'
gt_data_train_path = './processed_data/train/gt'
image_data_train_path = './processed_data/train/img'
density_data_train_path = './processed_data/train/den'

gt_data_val_path = './processed_data/val/gt'
image_data_val_path = './processed_data/val/img'
density_data_val_path = './processed_data/val/den'

gt_data_test_path = './processed_data/test/gt'
image_data_test_path = './processed_data/test/img'
density_data_test_path = './processed_data/test/den'

save_path = './models/model.pt'