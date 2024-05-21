import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt



class ImageProcessor:
    def __init__(self, master):
        self.master = master
        self.master.title("Görüntü İşleme Arayüzü")
        
        # Ana çerçeve ve butonları oluşturma
        self.frame = tk.Frame(master)
        self.frame.pack(side=tk.BOTTOM, pady=10)
        
        self.rotate_label = tk.Label(self.frame, text="Döndürme Açısı:")
        self.rotate_label.pack(side=tk.LEFT, padx=5)
        
        self.rotate_angle = tk.Scale(self.frame, from_=0, to=360, orient=tk.HORIZONTAL)
        self.rotate_angle.pack(side=tk.LEFT, padx=5)
        
        self.zoom_label = tk.Label(self.frame, text="Yakınlaştırma/Uzaklaştırma:")
        self.zoom_label.pack(side=tk.LEFT, padx=5)
        
        self.zoom_scale = tk.Scale(self.frame, from_=50, to=200, orient=tk.HORIZONTAL)
        self.zoom_scale.set(100)
        self.zoom_scale.pack(side=tk.LEFT, padx=5)
        
        
        self.low_threshold_scale = tk.Label(self.frame, text="Düşük Eşik:")
        self.low_threshold_scale.pack(side=tk.LEFT, padx=5)
        self.low_threshold_scale = tk.Scale(self.frame, from_=0, to=255, orient=tk.HORIZONTAL)
        self.low_threshold_scale.set(50)
        self.low_threshold_scale.pack(side=tk.LEFT, padx=5)
        
        self.high_threshold_scale = tk.Label(self.frame, text="Yüksek Eşik:")
        self.high_threshold_scale.pack(side=tk.LEFT, padx=5)
        self.high_threshold_scale = tk.Scale(self.frame, from_=0, to=255, orient=tk.HORIZONTAL)
        self.high_threshold_scale.set(150)
        self.high_threshold_scale.pack(side=tk.LEFT, padx=5)
        
        self.reduce_contrast_label = tk.Label(self.frame, text="Kontrast Azaltma Miktarı:")
        self.reduce_contrast_label.pack(side=tk.LEFT, padx=5)
        
        self.reduce_contrast_scale = tk.Scale(self.frame, from_=0.1, to=1.0, resolution=0.1, orient=tk.HORIZONTAL)
        self.reduce_contrast_scale.set(0.5)
        self.reduce_contrast_scale.pack(side=tk.LEFT, padx=5)
        
        self.noise_label = tk.Label(self.frame, text="Gürültü Oranı:")
        self.noise_label.pack(side=tk.LEFT, padx=5)
        
        self.noise_scale = tk.Scale(self.frame, from_=1, to=10, orient=tk.HORIZONTAL)
        self.noise_scale.set(5)
        self.noise_scale.pack(side=tk.LEFT, padx=5)
        
        self.motion_blur_label = tk.Label(self.frame, text="Motion Blur Uzunluğu:")
        self.motion_blur_label.pack(side=tk.LEFT, padx=5)
        
        self.motion_blur_scale = tk.Scale(self.frame, from_=1, to=50, orient=tk.HORIZONTAL)
        self.motion_blur_scale.set(10)
        self.motion_blur_scale.pack(side=tk.LEFT, padx=5)
        
        self.process_buttons_frame = tk.Frame(master)
        self.process_buttons_frame.pack(side=tk.TOP, pady=10)

        # Butonları içeren tablo
        self.process_buttons_table = tk.Frame(self.process_buttons_frame)
        self.process_buttons_table.pack()
        
        process_buttons_info = {
            "Dosya Seç": self.upload_image,
            "İkinci Dosya Seç": self.upload_second_image,
            "Gri Dönüşüm": self.grayscale_image,
            "Döndür": self.rotate_image,
            "Yakınlaştır/Uzaklaştır": self.zoom_image,
            "Binary Dönüşüm": self.binary_image,
            "Çift Eşikleme": self.double_threshold_image,
            "Median Filtre": self.median_filter,
            "Mean Filtre": self.mean_filter,
            "Kontrast Azalt": self.reduce_contrast,
            "Gürültü Ekle": self.add_salt_and_pepper_noise,
            "Genişletme": self.dilate_image,
            "Aşındırma": self.erode_image,
            "Açma": self.open_image,
            "Kapama": self.close_image,
            "Motion Blur": self.motion_blur_image,
            "Kırpma": self.start_crop,
            "Canny": self.canny_edge_detection,
            "Çarpma": self.multiply_images,
            "Çıkarma": self.subtract_images,
            "Histogram göster": self.show_histogram,
            "Histogram Germe": self.histogram_stretch,
            "RGB - HSV": self.convert_to_hsv,
            "RGB - LAB": self.convert_to_lab,
            "RGB - YCrCb": self.convert_to_ycrcb,
            
        }

    # Butonları tablolama   
        row_index = 0
        column_index = 0
        for button_text, button_command in process_buttons_info.items():
            button = tk.Button(self.process_buttons_table, text=button_text, command=button_command)
            button.grid(row=row_index, column=column_index, padx=5, pady=5)
            column_index += 1
            if column_index > 4:
                column_index = 0
                row_index += 1

        
        # Orijinal, ikinci ve işlenmiş görüntüleri göstermek için üç canvas oluşturma
        self.canvas_original_frame = tk.Frame(master)
        self.canvas_original_frame.pack(side=tk.LEFT, padx=10)
        self.canvas_original_label = tk.Label(self.canvas_original_frame, text="Birinci Resim")
        self.canvas_original_label.pack()
        self.canvas_original = tk.Canvas(self.canvas_original_frame, width=400, height=400, bg='white')
        self.canvas_original.pack()
        
        self.canvas_second_frame = tk.Frame(master)
        self.canvas_second_frame.pack(side=tk.LEFT, padx=10)
        self.canvas_second_label = tk.Label(self.canvas_second_frame, text="İkinci Resim")
        self.canvas_second_label.pack()
        self.canvas_second = tk.Canvas(self.canvas_second_frame, width=400, height=400, bg='white')
        self.canvas_second.pack()
        
        self.canvas_processed_frame = tk.Frame(master)
        self.canvas_processed_frame.pack(side=tk.LEFT, padx=10)
        self.canvas_processed_label = tk.Label(self.canvas_processed_frame, text="Sonuç")
        self.canvas_processed_label.pack()
        self.canvas_processed = tk.Canvas(self.canvas_processed_frame, width=400, height=400, bg='white')
        self.canvas_processed.pack()
        
        


        # En başta görüntü yolu boş 
        self.image = None
        self.image_path = None
        self.second_image = None

    def upload_image(self):
        # Görüntü seçme fonk
        self.image_path = filedialog.askopenfilename() 
        if self.image_path:
            self.image = cv2.imread(self.image_path) 
            if self.image is not None:
                self.image = self.resize_image(self.image, 400, 400)  
                self.display_image(self.image, self.canvas_original)  
            else:
                messagebox.showerror("Hata", "Görüntü yüklenemedi. Lütfen geçerli bir dosya seçin.")

    def upload_second_image(self):
        image_path = filedialog.askopenfilename()
        if image_path:
            self.second_image = cv2.imread(image_path)
            if self.second_image is not None:
                self.second_image = self.resize_image(self.second_image, 400, 400)
                self.display_image(self.second_image, self.canvas_second)
            else:
                messagebox.showerror("Hata", "Görüntü yüklenemedi. Lütfen geçerli bir dosya seçin.")
            

    def resize_image(self, image, max_width, max_height):
        
        height, width = image.shape[:2]
        if width > max_width or height > max_height:
            if width > height:
                new_width = max_width
                new_height = int(max_width * height / width)
            else:
                new_height = max_height
                new_width = int(max_height * width / height)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return image

    def display_image(self, image, canvas):
        canvas.delete("all")
        height, width = image.shape[:2]
        canvas.config(width=width, height=height)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image_tk = ImageTk.PhotoImage(image=image)
        canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
        canvas.image = image_tk

        # Çarpma
    def multiply_images(self):
        if self.image is not None and self.second_image is not None:
            first_image_resized, second_image_resized = self.match_image_sizes(self.image, self.second_image)
            result = cv2.multiply(first_image_resized, second_image_resized)
            self.display_image(result, self.canvas_processed)
        else:
            messagebox.showerror("Hata", "Lütfen iki resmi de yükleyin.")
    # Çıkartma
    def subtract_images(self):
        if self.image is not None and self.second_image is not None:
            first_image_resized, second_image_resized = self.match_image_sizes(self.image, self.second_image)
            result = cv2.subtract(first_image_resized, second_image_resized)
            self.display_image(result, self.canvas_processed)
        else:
            messagebox.showerror("Hata", "Lütfen iki resmi de yükleyin.")
    
    def match_image_sizes(self, img1, img2):
        height1, width1 = img1.shape[:2]
        height2, width2 = img2.shape[:2]
        max_width = max(width1, width2)
        max_height = max(height1, height2)
        img1_resized = cv2.resize(img1, (max_width, max_height), interpolation=cv2.INTER_AREA)
        img2_resized = cv2.resize(img2, (max_width, max_height), interpolation=cv2.INTER_AREA)
        return img1_resized, img2_resized

        
        # GRİ DÖNÜŞÜM
    def grayscale_image(self):
        if self.image is not None:
        
           gray_image = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)

        
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
              
                r, g, b = self.image[i, j]
                
                gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
                
                gray_image[i, j] = gray_value
        
        gray_image_colored = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
       
        self.display_image(gray_image_colored, self.canvas_processed)
       
        #DÖNDÜRME
    def rotate_image(self):
        if self.image is not None:  
            angle = self.rotate_angle.get() 
            (h, w) = self.image.shape[:2]
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_image = cv2.warpAffine(self.image, M, (w, h))
            self.display_image(rotated_image, self.canvas_processed)  
        
        #YAKINLAŞTIRMA - UZAKLAŞTIRMA
    def zoom_image(self):
        if self.image is not None:  
            scale_percent = self.zoom_scale.get()  
            width = int(self.image.shape[1] * scale_percent / 100)
            height = int(self.image.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized_image = cv2.resize(self.image, dim, interpolation=cv2.INTER_AREA)
            self.display_image(resized_image, self.canvas_processed)  
        
        #BİNARY
    def binary_image(self):
        if self.image is not None:
       
           gray_image = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
        binary_image = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)

        
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                r, g, b = self.image[i, j]

                gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
                gray_image[i, j] = gray_value

                threshold = 128
                if gray_value > threshold:
                    binary_image[i, j] = 255
                else:
                    binary_image[i, j] = 0

        binary_image_colored = cv2.merge([binary_image, binary_image, binary_image])
        
        self.display_image(binary_image_colored, self.canvas_processed)

        #ÇİFT EŞİKLEME
    def double_threshold_image(self):
        if self.image is not None:
            low_threshold = self.low_threshold_scale.get()  # Düşük eşik değeri
            high_threshold = self.high_threshold_scale.get()  # Yüksek eşik değeri
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            _, binary_low = cv2.threshold(gray_image, low_threshold, 255, cv2.THRESH_BINARY)
            _, binary_high = cv2.threshold(gray_image, high_threshold, 255, cv2.THRESH_BINARY_INV)
            double_thresholded_image = cv2.bitwise_and(binary_low, binary_high)
            double_thresholded_image = cv2.cvtColor(double_thresholded_image, cv2.COLOR_GRAY2BGR)
            self.display_image(double_thresholded_image, self.canvas_processed)  
    
        #MEDİAN
    def median_filter(self):
        if self.image is not None:
        
           filter_size = 3
        pad_size = filter_size // 2

        padded_image = np.pad(self.image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='edge')

        median_filtered_image = np.zeros_like(self.image)

        
        for i in range(pad_size, padded_image.shape[0] - pad_size):
            for j in range(pad_size, padded_image.shape[1] - pad_size):
                for c in range(self.image.shape[2]): 
                    
                    neighborhood = padded_image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1, c]
                    median_value = np.median(neighborhood)
                    median_filtered_image[i - pad_size, j - pad_size, c] = median_value

        self.display_image(median_filtered_image, self.canvas_processed)
        #MEAN
    def mean_filter(self):
        if self.image is not None:
       
           filter_size = 3
        pad_size = filter_size // 2

        padded_image = np.pad(self.image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='edge')

    
        mean_filtered_image = np.zeros_like(self.image)

        for i in range(pad_size, padded_image.shape[0] - pad_size):
            for j in range(pad_size, padded_image.shape[1] - pad_size):
                for c in range(self.image.shape[2]):  
                   
                    neighborhood = padded_image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1, c]

                   
                    mean_value = np.mean(neighborhood)

                   
                    mean_filtered_image[i - pad_size, j - pad_size, c] = mean_value

        
        self.display_image(mean_filtered_image, self.canvas_processed)
        #KONTRAST AZALTMA
    def reduce_contrast(self):
        if self.image is not None:
           amount = self.reduce_contrast_scale.get()  
           factor = amount / 10.0

       
        normalized_image = self.image.astype(np.float32) / 255.0

        mean_luminance = np.mean(normalized_image)

        reduced_contrast_image = mean_luminance + factor * (normalized_image - mean_luminance)

        reduced_contrast_image = np.clip(reduced_contrast_image * 255.0, 0, 255).astype(np.uint8)

        self.display_image(reduced_contrast_image, self.canvas_processed)
        #SALT PEPPER
    def add_salt_and_pepper_noise(self):
        if self.image is not None:
           amount = self.noise_scale.get() / 100.0  

        noisy_image = self.image.copy()
        num_salt = int(amount * noisy_image.size * 0.5)  
        num_pepper = int(amount * noisy_image.size * 0.5) 

        
        for _ in range(num_salt):
            i = np.random.randint(0, noisy_image.shape[0])
            j = np.random.randint(0, noisy_image.shape[1])
            noisy_image[i, j] = [255, 255, 255]

        
        for _ in range(num_pepper):
            i = np.random.randint(0, noisy_image.shape[0])
            j = np.random.randint(0, noisy_image.shape[1])
            noisy_image[i, j] = [0, 0, 0]

        self.display_image(noisy_image, self.canvas_processed)
        #MORFOLOJİK İŞLEMLER
        #GENİŞLETME
    def dilate_image(self):
        if self.image is not None:
    
         kernel_size = 5
         kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
         dilated_image = np.zeros_like(self.image)
         for c in range(self.image.shape[2]):  
             for i in range(self.image.shape[0] - kernel_size + 1):
                 for j in range(self.image.shape[1] - kernel_size + 1):
                    patch = self.image[i:i + kernel_size, j:j + kernel_size, c]
                    max_val = np.max(patch * kernel)
                    dilated_image[i + kernel_size // 2, j + kernel_size // 2, c] = max_val

         self.display_image(dilated_image, self.canvas_processed)
        else:
          messagebox.showerror("Hata", "Lütfen bir görüntü yükleyin.")
        #AŞINDIRMA
    def erode_image(self):
        if self.image is not None:
      
         kernel_size = 5
         kernel = np.ones((kernel_size, kernel_size), np.uint8)

         eroded_image = np.zeros_like(self.image)
         for c in range(self.image.shape[2]):  
             for i in range(self.image.shape[0] - kernel_size + 1):
                for j in range(self.image.shape[1] - kernel_size + 1):
                    patch = self.image[i:i + kernel_size, j:j + kernel_size, c]
                    min_val = np.min(patch * kernel)
                    eroded_image[i + kernel_size // 2, j + kernel_size // 2, c] = min_val

         self.display_image(eroded_image, self.canvas_processed)
        else:
         messagebox.showerror("Hata", "Lütfen bir görüntü yükleyin.")  
        
        #AÇMA
    def open_image(self):
        if self.image is not None:
       
          kernel_size = 5
          kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Erozyon işlemi
          eroded_image = np.zeros_like(self.image)
          for c in range(self.image.shape[2]):
            for i in range(self.image.shape[0] - kernel_size + 1):
                for j in range(self.image.shape[1] - kernel_size + 1):
                    patch = self.image[i:i + kernel_size, j:j + kernel_size, c]
                    min_val = np.min(patch * kernel)
                    eroded_image[i + kernel_size // 2, j + kernel_size // 2, c] = min_val

        # Genişletme işlemi
          opened_image = np.zeros_like(eroded_image)
          for c in range(eroded_image.shape[2]):
             for i in range(eroded_image.shape[0] - kernel_size + 1):
                for j in range(eroded_image.shape[1] - kernel_size + 1):
                    patch = eroded_image[i:i + kernel_size, j:j + kernel_size, c]
                    max_val = np.max(patch * kernel)
                    opened_image[i + kernel_size // 2, j + kernel_size // 2, c] = max_val

          self.display_image(opened_image, self.canvas_processed)
        else:
          messagebox.showerror("Hata", "Lütfen bir görüntü yükleyin.")
        #KAPAMA
    def close_image(self):
        if self.image is not None:
         kernel_size = 5
         kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Genişletme işlemi
         dilated_image = np.zeros_like(self.image)
         for c in range(self.image.shape[2]): 
            for i in range(self.image.shape[0] - kernel_size + 1):
                for j in range(self.image.shape[1] - kernel_size + 1):
                    patch = self.image[i:i + kernel_size, j:j + kernel_size, c]
                    max_val = np.max(patch * kernel)
                    dilated_image[i + kernel_size // 2, j + kernel_size // 2, c] = max_val

        # Erozyon işlemi
         closed_image = np.zeros_like(dilated_image)
         for c in range(dilated_image.shape[2]): 
            for i in range(dilated_image.shape[0] - kernel_size + 1):
                for j in range(dilated_image.shape[1] - kernel_size + 1):
                    patch = dilated_image[i:i + kernel_size, j:j + kernel_size, c]
                    min_val = np.min(patch * kernel)
                    closed_image[i + kernel_size // 2, j + kernel_size // 2, c] = min_val

         self.display_image(closed_image, self.canvas_processed)
        else:
         messagebox.showerror("Hata", "Lütfen bir görüntü yükleyin.")   
        #MOTİON
    def motion_blur_image(self):
        if self.image is not None:
            size = self.motion_blur_scale.get()
            kernel_motion_blur = np.zeros((size, size))
            kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
            kernel_motion_blur = kernel_motion_blur / size
            motion_blurred_image = cv2.filter2D(self.image, -1, kernel_motion_blur)
            self.display_image(motion_blurred_image, self.canvas_processed)  
        
        #HİSTOGRAM GÖSTERME
    def show_histogram(self):
        if self.image is not None:
            image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  
            plt.figure()
            plt.hist(image_gray.ravel(), bins=256, range=[0, 256])
            plt.title('Histogram')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.show()
        
        #HİSTOGRAM GERME GENİŞLETME
    def histogram_stretch(self): 
        if self.image is not None:
            image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            image_eq = cv2.equalizeHist(image_gray)  # Histogram eşitleme 
            image_eq = cv2.cvtColor(image_eq, cv2.COLOR_GRAY2BGR)  
            self.display_image(image_eq, self.canvas_processed)  
    
    # RENK UZAY DÖNÜŞÜMLERİ
    def convert_to_hsv(self):
        if self.image is not None:
        
           rgb_image = self.image

        hsv_image = np.zeros_like(rgb_image, dtype=np.float32)

        for i in range(rgb_image.shape[0]):
            for j in range(rgb_image.shape[1]):
                r = rgb_image[i, j, 0] / 255.0
                g = rgb_image[i, j, 1] / 255.0
                b = rgb_image[i, j, 2] / 255.0

                cmax = max(r, g, b)
                cmin = min(r, g, b)
                delta = cmax - cmin

                if delta == 0:
                    h = 0
                elif cmax == r:
                    h = 60 * (((g - b) / delta) % 6)
                elif cmax == g:
                    h = 60 * (((b - r) / delta) + 2)
                elif cmax == b:
                    h = 60 * (((r - g) / delta) + 4)

               
                hsv_image[i, j, 0] = h

                # Satürasyon
                if cmax == 0:
                    s = 0
                else:
                    s = delta / cmax
                hsv_image[i, j, 1] = s

                #değer 
                v = cmax
                hsv_image[i, j, 2] = v

        hsv_image = (hsv_image * 255).astype(np.uint8)

        self.display_image(hsv_image, self.canvas_processed)
    
    def convert_to_lab(self):
        if self.image is not None:
      
           rgb_image = self.image

        
        normalized_image = rgb_image.astype(np.float32) / 255.0

       
        def rgb_to_xyz(rgb):
           
            m = np.array([[0.4124564, 0.3575761, 0.1804375],
                          [0.2126729, 0.7151522, 0.0721750],
                          [0.0193339, 0.1191920, 0.9503041]])
            return np.dot(rgb, m.T)

        xyz_image = rgb_to_xyz(normalized_image)

      
        def xyz_to_lab(xyz):
           
            ref_xyz = np.array([0.950456, 1.0, 1.088754])

          
            def f(t):
                delta = 6 / 29
                if t > delta ** 3:
                    return t ** (1 / 3)
                else:
                    return t / (3 * delta ** 2) + 4 / 29

           
            f_xyz = np.vectorize(f)
            lab = np.zeros_like(xyz)
            lab[:, :, 0] = 116 * f_xyz(xyz[:, :, 1] / ref_xyz[1]) - 16
            lab[:, :, 1] = 500 * (f_xyz(xyz[:, :, 0] / ref_xyz[0]) - f_xyz(xyz[:, :, 1] / ref_xyz[1]))
            lab[:, :, 2] = 200 * (f_xyz(xyz[:, :, 1] / ref_xyz[1]) - f_xyz(xyz[:, :, 2] / ref_xyz[2]))
            return lab

        lab_image = xyz_to_lab(xyz_image)

        lab_image[:, :, 0] = np.clip(lab_image[:, :, 0], 0, 100)  
        lab_image[:, :, 1:] = np.clip(lab_image[:, :, 1:], -128, 127)  

        
        lab_image = ((lab_image + [0, 128, 128]) / [100, 255, 255] * 255).astype(np.uint8)

        self.display_image(lab_image, self.canvas_processed)

    def convert_to_ycrcb(self):
        if self.image is not None:
        
           rgb_image = self.image

        
        normalized_image = rgb_image.astype(np.float32) / 255.0

       
        def rgb_to_ycrcb(rgb):
          
            m = np.array([[65.481, 128.553, 24.966],
                          [-37.797, -74.203, 112.0],
                          [112.0, -93.786, -18.214]])
            return np.dot(rgb, m.T)

        ycbcr_image = rgb_to_ycrcb(normalized_image)

      
        ycbcr_image[:, :, 0] = np.clip(ycbcr_image[:, :, 0], 16, 235) 
        ycbcr_image[:, :, 1:] = np.clip(ycbcr_image[:, :, 1:], 16, 240) 

  
        ycbcr_image = ((ycbcr_image + [0, 128, 128]) / [255, 255, 255] * 255).astype(np.uint8)

        self.display_image(ycbcr_image, self.canvas_processed)


        #KIRPMA
    def start_crop(self):
        messagebox.showinfo("Kırpma", "Lütfen kırpma alanını seçin ve Enter tuşuna basın.")
        self.master.bind('<Button-1>', self.get_mouse_click)
        self.master.bind('<Return>', self.crop_image)

    def get_mouse_click(self, event):
        self.start_x, self.start_y = event.x, event.y

    def crop_image(self, event):
        self.master.unbind('<Button-1>')
        self.master.unbind('<Return>')
        if self.image is not None:
            end_x, end_y = event.x, event.y
            cropped = self.image[self.start_y:end_y, self.start_x:end_x]
            self.display_image(cropped, self.canvas_processed)
    #CANNY
    def canny_edge_detection(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            self.display_image(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), self.canvas_processed)
       
    

#  Tkinter penceresi olş. genel görüntüleme
root = tk.Tk()
app = ImageProcessor(root)
root.mainloop()
