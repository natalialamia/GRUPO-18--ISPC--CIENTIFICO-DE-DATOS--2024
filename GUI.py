import tkinter as tk
from tkinter import filedialog, ttk
from PIL import ImageTk, Image
import mysql.connector
import os
from shutil import copy2
import io
import numpy as np

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.image as mpimg  # For loading images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
ima = mpimg.imread("uno.jpg")
from pygame import mixer, time
import os
import importlib


class ImageViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Turismo")
        self.geometry("900x600")      
        self.images = []
        self.current_image_index = 0
        self.recommended_images = []
        self.photo_image = None
        self.current_image_index = 0
        self.table_name=""
        self.images = []
        self.current_image_index = 0
        self.recommended_images = []
        self.photo_image = None
        self.current_image_index = 0
        self.p=0
        self.imagenes_module = None
        self.canvas_derecho = None
    
        self.canvas_derecho = None  # Inicializar canvas_derecho como None
# Colores para los paneles
        self.color_panel_superior = "blue"
        self.color_panel_inferior = "blue"
        self.color_panel_izquierdo = ""
        self.color_panel_derecho = ""

  # Panel superior (10% de la ventana)
        self.panel_superior = tk.Frame(self, height=60)  
        self.panel_superior.pack(side=tk.TOP, fill=tk.X)
        self.hacer_transparente(self.panel_superior, alpha=0.7)
     
        
    # Panel inferior (10% de la ventana)
        self.panel_inferior = tk.Frame(self, bg=self.color_panel_inferior, height=60)
        self.panel_inferior.pack(side=tk.BOTTOM, fill=tk.X)

   # Panel central dividido en izquierda y derecha
        self.panel_central = tk.Frame(self)
        self.panel_central.pack(fill=tk.BOTH, expand=True)

    # Panel izquierdo
        self.panel_izquierdo = tk.Frame(self.panel_central,bg=self.color_panel_izquierdo)
        self.panel_izquierdo.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

    # Canvas en el panel izquierdo
        self.canvas_izquierdo = tk.Canvas(self.panel_izquierdo,bg="white")
        self.canvas_izquierdo.pack(fill=tk.BOTH, padx=20, pady=20,expand=True)
        
       
 # Frame con botones y etiqueta en el panel izquierdo
        frame_botones_izquierdo = tk.Frame(self.panel_izquierdo, bg=self.color_panel_izquierdo)
        frame_botones_izquierdo.pack(side=tk.BOTTOM, fill=tk.X)
        boton1 = tk.Button(frame_botones_izquierdo, text="Cargar Imagen", command=self.open_image)
        boton1.pack(side=tk.LEFT)  
        self.info_label2 = tk.Label(frame_botones_izquierdo, text="  ",background='#f0dd4e')  
        self.info_label2.pack(side=tk.LEFT)
             
        boton2 = tk.Button(frame_botones_izquierdo, text="Predecir", command=self.predict_image_and_update_label)
        boton2.pack(side=tk.LEFT)
        self.info_label3 = tk.Label(frame_botones_izquierdo, text="  ",background='#f0dd4e')  
        self.info_label3.pack(side=tk.LEFT)
        boton3 = tk.Button(frame_botones_izquierdo, text="Escuchar", command=self.reproducir2)
        boton3.pack(side=tk.LEFT)        	
        self.info_label4 = tk.Label(frame_botones_izquierdo, text="  ",background='#f0dd4e') 
        self.info_label4.pack(side=tk.LEFT)
        self.info_label = tk.Label(frame_botones_izquierdo, text="---prediccion---",background='#fddd4d',font=("Helvetica", 18, "bold"))  
        self.info_label.pack(side=tk.LEFT)
        


 # Panel derecho
        self.panel_derecho = tk.Frame(self.panel_central, bg=self.color_panel_derecho)
        self.panel_derecho.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        

    # Canvas en el panel derecho
      #  self.canvas_derecho = tk.Canvas(self.panel_derecho, bg="red")
      #  self.canvas_derecho.pack(fill=tk.BOTH,padx=20, pady=20, expand=True)


    # Frame con botones en el panel derecho
        frame_botones_derecho = tk.Frame(self.panel_derecho, bg=self.color_panel_derecho)
        frame_botones_derecho.pack(side=tk.BOTTOM, fill=tk.X)
        boton4 = tk.Button(frame_botones_derecho, text="Recomendar", command=self.recommend_images)
        boton4.pack(side=tk.LEFT)
        #boton5 = tk.Button(frame_botones_derecho, text="Botón 5")
        #boton5.pack(side=tk.LEFT)
        #boton6 = tk.Button(frame_botones_derecho, text="Botón 6")
        #boton6.pack(side=tk.LEFT)

        self.hacer_transparente(self.panel_inferior , alpha=0)
        self.hacer_transparente(self.panel_superior , alpha=0)


       # Conexión a la base de datos MySQL
        self.db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="mariano",
            database="recomendaciones"
        )
        self.cursor = self.db.cursor()
  
    def hacer_transparente(self, frame, alpha=0.5):
        canvas = tk.Canvas(frame, bg="white", highlightthickness=0)
        canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        canvas.create_rectangle(0, 0, frame.winfo_width(), frame.winfo_height(), fill="", outline="", stipple="gray50")

   
    

     
#################

#####################
     

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            destination_folder = r"C:\Users\Usuario\Desktop\archivos\final gui domingo\img_subida\img_subida"
             # Crear la carpeta de destino si no existe
            if not os.path.exists(destination_folder):
               os.makedirs(destination_folder)

           # Obtener el nombre de archivo
            filename = os.path.basename(file_path)
            print (filename)	
            filename="imagen1.jpg"

           # Construir la ruta completa de destino
            destination_path = os.path.join(destination_folder, filename)

          # Copiar la imagen a la carpeta de destino
            copy2(file_path, destination_path)

            image = Image.open(file_path)
            image = image.resize((400, 400))
            self.images.append(image)
            self.show_image(len(self.images) - 1)
            self.current_image_index = len(self.images) - 1
            self.update_buttons()

    def show_image(self, index):
        self.canvas_izquierdo.delete("all")
        image = self.images[index]
        photo_image = ImageTk.PhotoImage(image)
        self.canvas_izquierdo.create_image( 200,200,image=photo_image)
        self.canvas_izquierdo.image = photo_image

    def reproducir(self):
        mixer.init()
        mixer.music.set_volume(1)
	
        # Obtener la fecha y hora de modificación del archivo
        audio_file_path = "result_audio.mp3"
        file_modified_time = os.path.getmtime(audio_file_path)

       # Comparar la fecha y hora de modificación con una variable de instancia
        #if file_modified_time != getattr(self, "audio_file_modified_time", None):
       	 # El archivo ha sido actualizado, cargar y reproducir
        mixer.music.load(audio_file_path)
        mixer.music.play()
       # self.audio_file_modified_time = file_modified_time
        
    def reproducir2(self):
        mixer.quit()
        self.reproducir()
   
   
    def predict_image_and_update_label(self):
        mixer.quit()
        if self.imagenes_module is None:
            import imagenes
            self.imagenes_module = imagenes
            print("Predecir imagen:", self.images[self.current_image_index])
            nuevo_texto = self.imagenes_module.class_name   
            print(self.imagenes_module.class_name)
            self.info_label.config(text=nuevo_texto)
        else:
            importlib.reload(self.imagenes_module)
            
            print("Predecir imagen:", self.images[self.current_image_index])
            nuevo_texto = self.imagenes_module.class_name
            print(self.imagenes_module.class_name)
            self.info_label.config(text=nuevo_texto)
        self.table_name = self.imagenes_module.class_name  
        # Limpiar los subplots antiguos
        if hasattr(self, 'plots'):
            for plot in self.plots:
                plot.clear()
        else:
            self.plots = []        


    def predict_image(self):
        self.predict_image_and_update_label()
        #import imagenes
        #print("Predecir imagen:", self.images[self.current_image_index])
        #nuevo_texto = imagenes.class_name
        #print(imagenes.class_name)
        #self.info_label1.config(text="      ")
        #self.info_label.config(text=nuevo_texto)
 

    def clear_canvas_derecho(self):
        if self.canvas_derecho:
            self.canvas_derecho.get_tk_widget().delete("all")

    def recommend_images(self):
        self.clear_canvas_derecho()  # Limpiar el canvas_derecho antes de dibujar

        num_images = 4
        self.recommended_images = []

      # Subplot creation and configuration
         # Define frame dimensions
        frame_width = 100
        frame_height = 100

        # Create the Matplotlib figure
        fig = Figure(figsize=(4, 3))  # Adjust figure size as needed

        # Subplot creation and configuration
        a = fig.add_subplot(221) 
        b = fig.add_subplot(222)
        c = fig.add_subplot(223)
        d = fig.add_subplot(224)
          # Limpiar los subplots existentes
        a.clear()
        b.clear()
        c.clear()
        d.clear() 
# Agregar las nuevas referencias de subplots a self.plots
        self.plots = [a, b, c, d]
        if self.canvas_derecho:
            self.canvas_derecho.get_tk_widget().pack_forget()

        if self.table_name == "forest":
             # Obtener imágenes de la base de datos
            self.cursor.execute("SELECT paisajes FROM bosques LIMIT %s", (num_images,))
            resultados = self.cursor.fetchall()
            self.clear_canvas_derecho()
            for i, row in enumerate(resultados):
                image_data = row[0]
                image = ImageTk.PhotoImage(data=image_data)
                #self.recommended_images.append(image)
                
                # Convertir la imagen a una matriz de píxeles
                image_pil = Image.open(io.BytesIO(image_data))
                image_array = np.asarray(image_pil)

               # Mostrar la imagen en el subplot correspondiente
                if i == 0:
                    a.imshow(image_array)
                    a.axis("off")
                    a.set_title("bosque andino")
                elif i == 1:
                    b.imshow(image_array)
                    b.axis("off")
                    b.set_title("selva misionera")
                elif i == 2:
                    c.imshow(image_array)
                    c.axis("off")
                    c.set_title("bosque chaqueño")
                elif i == 3:
                    d.imshow(image_array)
                    d.axis("off")
                    d.set_title("tigre bsas")

        elif self.table_name == "street":
           # Obtener imágenes de la base de datos
            self.cursor.execute("SELECT paisajes FROM calles LIMIT %s", (num_images,))
            resultados = self.cursor.fetchall()
            
            for i, row in enumerate(resultados):
                image_data = row[0]
                image = ImageTk.PhotoImage(data=image_data)
                #self.recommended_images.append(image)
                
                # Convertir la imagen a una matriz de píxeles
                image_pil = Image.open(io.BytesIO(image_data))
                image_array = np.asarray(image_pil)

               # Mostrar la imagen en el subplot correspondiente
                if i == 0:
                    a.imshow(image_array)
                    a.axis("off")
                    a.set_title("CABA-san martin")
                elif i == 1:
                    b.imshow(image_array)
                    b.axis("off")
                    b.set_title("Calle reconquista")
                elif i == 2:
                    c.imshow(image_array)
                    c.axis("off")
                    c.set_title("Av sabattini")
                elif i == 3:
                    d.imshow(image_array)
                    d.axis("off")
                    d.set_title("s.martin-rosario")
        elif self.table_name == "sea":
           # Obtener imágenes de la base de datos
            self.cursor.execute("SELECT paisajes FROM mar LIMIT %s", (num_images,))
            resultados = self.cursor.fetchall()
            
            for i, row in enumerate(resultados):
                image_data = row[0]
                image = ImageTk.PhotoImage(data=image_data)
                #self.recommended_images.append(image)
                
                # Convertir la imagen a una matriz de píxeles
                image_pil = Image.open(io.BytesIO(image_data))
                image_array = np.asarray(image_pil)

               # Mostrar la imagen en el subplot correspondiente
                if i == 0:
                    a.imshow(image_array)
                    a.axis("off")
                    a.set_title("gesell")
                elif i == 1:
                    b.imshow(image_array)
                    b.axis("off")
                    b.set_title("mar del plata")
                elif i == 2:
                    c.imshow(image_array)
                    c.axis("off")
                    c.set_title("miramar")
                elif i == 3:
                    d.imshow(image_array)
                    d.axis("off")
                    d.set_title("necochea")
        elif self.table_name == "mountain":
             # Obtener imágenes de la base de datos
            self.cursor.execute("SELECT paisajes FROM montana LIMIT %s", (num_images,))
            resultados = self.cursor.fetchall()
            
            for i, row in enumerate(resultados):
                image_data = row[0]
                image = ImageTk.PhotoImage(data=image_data)
                #self.recommended_images.append(image)
                
                # Convertir la imagen a una matriz de píxeles
                image_pil = Image.open(io.BytesIO(image_data))
                image_array = np.asarray(image_pil)

               # Mostrar la imagen en el subplot correspondiente
                if i == 0:
                    a.imshow(image_array)
                    a.axis("off")
                    a.set_title("aconcagua")
                elif i == 1:
                    b.imshow(image_array)
                    b.axis("off")
                    b.set_title("cba-nono")
                elif i == 2:
                    c.imshow(image_array)
                    c.axis("off")
                    c.set_title("cerro uritorco")
                elif i == 3:
                    d.imshow(image_array)
                    d.axis("off")
                    d.set_title("chilecito")

        elif self.table_name == "buildings":
           # Obtener imágenes de la base de datos
            self.cursor.execute("SELECT paisajes FROM edificios LIMIT %s", (num_images,))
            resultados = self.cursor.fetchall()
            
            for i, row in enumerate(resultados):
                image_data = row[0]
                image = ImageTk.PhotoImage(data=image_data)
                #self.recommended_images.append(image)
                
                # Convertir la imagen a una matriz de píxeles
                image_pil = Image.open(io.BytesIO(image_data))
                image_array = np.asarray(image_pil)

               # Mostrar la imagen en el subplot correspondiente
                if i == 0:
                    a.imshow(image_array)
                    a.axis("off")
                    a.set_title("alvear tower -bsas")
                elif i == 1:
                    b.imshow(image_array)
                    b.axis("off")
                    b.set_title("Torre Angela-cba")
                elif i == 2:
                    c.imshow(image_array)
                    c.axis("off")
                    c.set_title("Torre Dorrego-bs")
                elif i == 3:
                    d.imshow(image_array)
                    d.axis("off")
                    d.set_title("Torre Maui-rosario")

        elif self.table_name == "glacier":
            # Obtener imágenes de la base de datos
            self.cursor.execute("SELECT paisajes FROM glaciar LIMIT %s", (num_images,))
            resultados = self.cursor.fetchall()
            
            for i, row in enumerate(resultados):
                image_data = row[0]
                image = ImageTk.PhotoImage(data=image_data)
                #self.recommended_images.append(image)
                
                # Convertir la imagen a una matriz de píxeles
                image_pil = Image.open(io.BytesIO(image_data))
                image_array = np.asarray(image_pil)

               # Mostrar la imagen en el subplot correspondiente
                if i == 0:
                    a.imshow(image_array)
                    a.axis("off")
                    a.set_title("cerro mayo")
                elif i == 1:
                    b.imshow(image_array)
                    b.axis("off")
                    b.set_title("glaciar viedma")
                elif i == 2:
                    c.imshow(image_array)
                    c.axis("off")
                    c.set_title("perito moreno")
                elif i == 3:
                    d.imshow(image_array)
                    d.axis("off")
                    d.set_title("upsala")

        # Embed the figure in the Tkinter window
        self.canvas_derecho = FigureCanvasTkAgg(fig, master=self.panel_derecho)
        self.canvas_derecho.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas_derecho.draw()

    
if __name__ == "__main__":
    app = ImageViewer()
    app.mainloop()