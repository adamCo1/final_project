import numpy as np
import os
import tkinter as tk
from tkinter import filedialog as fd
from enum import Enum
import csv

# def predict_from_folder(model,images_to_predict_path):
#     img_size = 128
#     images = []
#     class flowers(Enum):
#         daisy = 0
#         dandelion = 1
#         rose = 2
#         sunflower = 3
#         tulip = 4
#     csv_intial_row = [['ImageName','Class']]
#     with open ('results.csv','w') as f:
#         writer = csv.writer(f)
#         writer.writerows(csv_intial_row)
#     for img in os.listdir(images_to_predict_path):
#         img = image.load_img(os.path.join(images_to_predict_path,img), target_size=(img_size, img_size))
#         img = image.img_to_array(img)
#         img = np.expand_dims(img, axis=0)
#         images.append(img)
#
#     images = np.vstack(images)
#     classes = model.predict_classes(images, batch_size=20)
#     predictions = []
#     predictions.append("Results: ")
#     for img, classed in zip(os.listdir(images_to_predict_path),classes):
#         with open ('results.csv','a') as f:
#             classified = flowers(classed).name
#             csv_row = [[img, classified]]
#             predictions.append(img +" : "+classified)
#             print(csv_row)
#             writer = csv.writer(f)
#             writer.writerows(csv_row)
#     return predictions
#
#


# def prepare_network_from_file(model_path):
#     model = load_model(model_path)
#     return model

#
# a=""
# str1 = "e"
class Browse(tk.Frame):
    """ Creates a frame that contains a button when clicked lets the user to select
    a file and put its filepath into an entry.
    """


    def __init__(self, master, initialdir='', filetypes=()):
        super().__init__(master)
        self.filepath = tk.StringVar()
        self.modelPath = tk.StringVar()
        self._initaldir = initialdir
        self._filetypes = filetypes
        self._create_widgets()
        self._display_widgets()

    def _create_widgets(self):
        self._entry_prediction = tk.Entry(self, textvariable=self.filepath, font=("bold", 10))
        a=self._entry_prediction
        self._entry_model = tk.Entry(self, textvariable = self.modelPath, font = ("bold", 10))
        self._button_prediction = tk.Button(self, text="Browse Model File",bg="blue",fg="white", command=self.browsePredict)
        self._button_model = tk.Button(self, text="Browse Images Folder",bg="blue",fg="white", command=self.browseModel)
        self._classify=tk.Button(self,text="Classify",bg="blue",fg="white", command=self.classify)
        self._label=tk.Label(self, text="Flower Classification.", bg="blue", fg="black",height=3, font=("bold", 14))



    def _display_widgets(self):

        self._label.pack(fill='y')
        self._button_model.pack(fill='y')
        self._entry_prediction.pack(fill='x', expand=True)
        self._button_prediction.pack(fill='y')
        self._entry_model.pack(fill='x', expand=True)
        self._classify.pack(fill='y')

    def retrieve_input(self):
        #str1 = self._entry.get()
        # a=a.replace('/','//')
        print (str1)


    def classify(self):
        newwin = tk.Toplevel(root)
        newwin.geometry("500x500")
        label = tk.Label(newwin, text="Classification", bg="blue", fg="white",height=3, font=("bold", 14))
        label.pack()
        model = prepare_network_from_file(self.modelPath.get())
        predictions = predict_from_folder(model,self.filepath.get())

        T = tk.Text(newwin, height=25, width=60)

        T.insert('end','\n'.join(predictions))
        T.pack()

        #then show it on screen
        newwin.mainloop()

    def browseModel(self):

        self.filepath.set(fd.askdirectory())

    def browsePredict(self):
        self.modelPath.set(fd.askopenfilename(initialdir=self._initaldir,
                                             filetypes=self._filetypes))


if __name__ == '__main__':
    root = tk.Tk()
    labelfont = ('times', 10, 'bold')
    root.geometry("500x500")
    filetypes = (
        ('Image File', '*.jpg'),
        ('Model File', '*.h5'),
        ("All files", "*.*")
    )

    file_browser = Browse(root, initialdir="\\",
                          filetypes=filetypes)
    file_browser.pack(fill='y')
    root.mainloop()