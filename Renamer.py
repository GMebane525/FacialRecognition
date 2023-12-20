import os
folder_path = r'C:\Users\gilme\OneDrive\Desktop\gr9-11'
for filename in os.listdir(folder_path):
   if filename.endswith('.jpg'):  # Change the extension according to your image format
       file_name, file_extension = os.path.splitext(filename)
       new_name = "'" + file_name[:-3].replace('_', '') + file_extension
       os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))